# early_regime_screener.py (Version 8.5 - AI Academy & Review Edition)
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# --- 設定ロード ---
CONFIG_FILE = Path("config.yaml")

def load_config() -> dict:
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

config = load_config()

# 環境変数
GMAIL_USER = os.environ.get("GMAIL_USER") or config.get("email", {}).get("gmail_user")
GMAIL_PASS = os.environ.get("GMAIL_APP_PASSWORD") or config.get("email", {}).get("gmail_pass")
NOTIFICATION_EMAIL = os.environ.get("NOTIFICATION_EMAIL") or config.get("email", {}).get("notification_email")
SENDER_NAME = "Sniper OS - AI Academy 8.5"

UNIVERSE_CSV = Path("universe.csv")
PRICES_DIR = Path("data_cache/prices")

# 設定ファイルから売買代金しきい値（デフォルト1,000万円）をロード
TH_MIN_TURNOVER = config.get("thresholds", {}).get("min_daily_turnover_million", 10.0)


def normalize_ticker(raw: str) -> str:
    ticker = str(raw).strip().upper()
    if not ticker:
        return ticker
    if "." not in ticker and not ticker.isdigit():
        ticker = f"{ticker}.T"
    return ticker


class EarlyStateEngine:
    @staticmethod
    def calculate_early_indicators(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["ma25"] = d["Close"].rolling(25).mean()
        d["ma75"] = d["Close"].rolling(75).mean()
        d["ma200"] = d["Close"].rolling(200).mean()
        
        # 25日線の傾き（直近5日比）
        d["ma25_slope"] = d["ma25"].pct_change(5) * 100
        d["ma75_slope"] = d["ma75"].pct_change(5) * 100
        d["ma200_slope"] = d["ma200"].pct_change(5) * 100
        d["ma25_slope_prev"] = d["ma25_slope"].shift(1)
        
        # 3本MA of 収縮幅（密集度）
        d["ma_max"] = d[["ma25", "ma75", "ma200"]].max(axis=1)
        d["ma_min"] = d[["ma25", "ma75", "ma200"]].min(axis=1)
        d["ma_mean"] = d[["ma25", "ma75", "ma200"]].mean(axis=1)
        d["ma_congestion_width_pct"] = (d["ma_max"] - d["ma_min"]) / d["ma_mean"] * 100
        
        # 収縮継続日数の動的カウント
        d["is_congested"] = d["ma_congestion_width_pct"] <= 5.0
        d["congestion_duration"] = d["is_congested"].groupby((~d["is_congested"]).cumsum()).cumsum()
        
        # 出来高比率
        d["vol_avg20"] = d["Volume"].rolling(20).mean()
        d["vol_ratio_20"] = d["Volume"] / d["vol_avg20"]
        d["turnover_avg20_million"] = ((d["Close"] * d["Volume"]) / 1_000_000).rolling(20).mean()
        
        # ボラティリティ
        std20 = d["Close"].rolling(20).std()
        d["bb_width"] = (std20 * 4) / d["ma25"] * 100
        
        # RSI
        delta = d["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        d["rsi14"] = 100 - (100 / (1 + (gain / np.where(loss > 0, loss, 1.0))))
        
        d["high_52w"] = d["High"].rolling(250, min_periods=50).max()
        d["dist_to_52w_high"] = (d["Close"] - d["high_52w"]) / d["high_52w"] * 100

        # Change Detection 用の Compression Score 計算（全行に適用）
        scores = []
        for idx, row in d.iterrows():
            score = 0
            width = row["ma_congestion_width_pct"]
            duration = row["congestion_duration"] if not pd.isna(row["congestion_duration"]) else 0
            bb_width = row["bb_width"]
            
            if width <= 1.5: score += 40
            elif width <= 3.0: score += 30
            elif width <= 5.0: score += 20
            else: score += 10
            
            if duration >= 25: score += 30
            elif duration >= 15: score += 20
            elif duration >= 5: score += 10
            else: score += 5
            
            if bb_width <= 5.0: score += 30
            elif bb_width <= 8.0: score += 20
            elif bb_width <= 12.0: score += 10
            else: score += 5
            scores.append(score)
        d["compression_score"] = scores

        return d


class EducationalAnalyzer:
    @staticmethod
    def get_ma_slope_symbols(latest_row: pd.Series) -> tuple[str, str, str]:
        def get_symbol(slope):
            if pd.isna(slope): return "→"
            if slope > 0.1: return "↑"
            elif slope < -0.1: return "↓"
            else: return "→"
        s25 = get_symbol(latest_row["ma25_slope"])
        s75 = get_symbol(latest_row["ma75_slope"])
        s200 = get_symbol(latest_row["ma200_slope"])
        return s25, s75, s200

    @staticmethod
    def get_trend_stage(latest_row: pd.Series, s25: str, s75: str, s200: str) -> str:
        close = latest_row["Close"]
        ma25 = latest_row["ma25"]
        ma75 = latest_row["ma75"]
        ma200 = latest_row["ma200"]
        if close > ma25 > ma75 > ma200:
            if s25 == "↑" and latest_row["ma25_slope_prev"] <= 0.1:
                return "◎ 上昇初期 (ブレイク直後の新鮮なトレンド)"
            return "○ 上昇途中 (強固な上昇トレンドを維持中)"
        elif close < ma25 < ma75 < ma200:
            return "× 下降継続 (典型的なダウントレンド継続中)"
        elif s25 == "→" and s75 == "→" and latest_row["ma_congestion_width_pct"] <= 3.5:
            return "△ 横ばい (エネルギー極限充填・充電中)"
        else:
            return "▲ 下降途中・もみ合い (底打ち転換の模索局面)"

    @staticmethod
    def get_expansion_readiness(latest_row: pd.Series, s25: str, s75: str, s200: str) -> str:
        close = latest_row["Close"]
        ma25 = latest_row["ma25"]
        ma75 = latest_row["ma75"]
        ma200 = latest_row["ma200"]
        if s25 == "↑" and s75 in ["↑", "→"] and close > ma75:
            return "S (極上：本上抜けカウントダウン状態)"
        elif s25 == "↑" and close > ma25:
            return "A (良好：トレンド発生の兆候あり)"
        elif s25 == "→" and latest_row["ma_congestion_width_pct"] <= 5.0:
            return "B (待機：エネルギー充填中で、反転のきっかけ待ち)"
        else:
            return "C (未成熟：底固めの途上にあり、点火には数日〜数週間が必要)"

    @staticmethod
    def generate_educational_comment(latest_row: pd.Series, trend_stage: str, pattern: str) -> str:
        """
        ②：完全動的な個別チャート着眼点生成エンジン
        """
        close = latest_row["Close"]
        ma25 = latest_row["ma25"]
        ma75 = latest_row["ma75"]
        ma200 = latest_row["ma200"]
        m25_slope = latest_row["ma25_slope"]
        m200_slope = latest_row["ma200_slope"]
        width = latest_row["ma_congestion_width_pct"]
        vol_ratio = latest_row["vol_ratio_20"]
        duration = int(latest_row["congestion_duration"]) if not pd.isna(latest_row["congestion_duration"]) else 0
        dev75 = abs(close - ma75) / ma75 * 100 if ma75 > 0 else 0.0

        if m200_slope < -0.15 and (close >= ma200 or abs(close - ma200)/ma200 <= 0.02):
            return (
                f"長期的に下降トレンドの蓋として機能していた『長期200日線（{ma200:.1f}円）』をブレイクできるかどうかの極めて重要な転換点です。 "
                f"200日線に頭を抑えられて押し返されるダマシになるか、それとも出来高を伴って本上抜けへ昇華するか、大口の介入意志を注視してください。"
            )
        elif dev75 <= 1.2 and ma75 >= ma200:
            return (
                f"上向きの中期トレンドラインである『75日移動平均線（{ma75:.1f}円）』に対して、株価がわずか {dev75:.2f}% と完全近接しています。 "
                f"ここが絶好の押し目（下値支持帯）として機能し、出来高の極限減少（売り枯れ）を経て、上方向への反発動意（買い手の出現）が起きるかを観察するのに最高の教材です。"
            )
        elif width <= 1.5 and duration >= 15:
            return (
                f"3本の移動平均線がわずか {width:.2f}% という極限レベルで密集し、その膠着が 【 {duration}営業日 】 も継続しています。 "
                f"煮詰まりは極限に達しており、直近のボックス上限をブレイクする際に、出来高が2〜3倍以上に『再点火』するかどうかを毎日観察してください。"
            )
        elif "ボックス" in pattern and duration >= 15:
            return (
                f"『{pattern}』というレンジ内で、売りと買いのバランスが均衡しています。 "
                f"出来高比率が {vol_ratio:.2f}倍 まで著しく細っているのは、投げ売る個人が全員いなくなった『売り枯れの呼吸』です。この極限状態からどちらに拡散エネルギーが放たれるかを定点観測してください。"
            )
        elif m25_slope > 0.3 and close > ma25:
            return (
                f"短期25日移動平均線が【上向き】へ反転し、ローソク足がその上に乗る新鮮な上昇動意の形です。 "
                f"ここから上放れが本格化する際、上方に存在する長期200日線（抵抗帯）とぶつかった時の押し引きを先行的に脳内シミュレーションしておきましょう。"
            )
        else:
            return (
                f"移動平均線の密集度が極めて狭い、典型的な『エネルギーの限界充填期』です。 "
                f"下値支持帯である75日移動平均線付近で株価がしっかりと下げ止まり、出来高が消滅する需給の真空状態をじっくり観察してください。"
            )

    @staticmethod
    def generate_chart_checklist(latest_row: pd.Series) -> list[str]:
        """
        ②：チャート観察チェックリスト（今日見る3つのポイント）
        """
        close = latest_row["Close"]
        ma75 = latest_row["ma75"] if "ma75" in latest_row else close
        vol_avg = latest_row["vol_avg20"] if "vol_avg20" in latest_row else 1000
        
        points = [
            f"□ 終値が支持帯（75日線: {ma75:.1f}円）を完全にキープし、下げ止まりを維持できるか？",
            f"□ 明日の出来高が、本日の出来高（{latest_row['Volume']:.0f}株）および20日平均（{vol_avg:.0f}株）を超えて再点火してくるか？",
            f"□ ローソク足の実体が陽線となり、短期25日移動平均線の『上向きの角度』を維持できるか？"
        ]
        return points

    @staticmethod
    def generate_tomorrow_mission(latest_row: pd.Series) -> str:
        """
        ③：明日見るポイント (Tomorrow's Mission)
        """
        vol = float(latest_row["vol_ratio_20"]) if "vol_ratio_20" in latest_row else 1.0
        bb = float(latest_row["bb_width"]) if "bb_width" in latest_row else 10.0
        
        mission = "【Tomorrow's Mission（明日の観察任務）】\n"
        if vol <= 0.60 and bb <= 5.0:
            mission += "  ➔ 明日は『売り枯れの極限（出来高0.6倍以下）からの大口の買い点火（出来高2倍以上）』を待ち伏せる日です。もし出来高が突如急増した場合は、授業Lesson 10の法則が始動したと判断します。"
        elif vol > 1.2:
            mission += "  ➔ 明日は『ふくらんだ出来高がスッと収縮に移行するか（売り枯れの初期段階）』を観察します。出来高が半分以下に細りながら下げ止まるかを注視してください。"
        else:
            mission += "  ➔ 明日は『移動平均密集エリア（3MA密集）のねじれ合いの継続』を観察します。価格が移動平均の束の上で推移し、下値が完全に支えられているかを確認してください。"
        return mission


def get_curriculum_lesson(date_str: str) -> dict:
    """
    ①：Today's Lesson のカリキュラム化 (30日サイクルでの体系的授業)
    年初からの経過日数に基づいて、毎日体系的な1テーマを提示
    """
    try:
        yday = datetime.strptime(date_str, "%Y-%m-%d").timetuple().tm_yday
    except Exception:
        yday = 1
        
    curriculum_cycle = {
        0: {
            "title": "Lesson 1: 移動平均線（MA）の基本原則と大衆コストの収束",
            "desc": "移動平均線は、過去の一定期間における投資家の『平均取得単価（コスト）』の推移を表します。25日・75日・200日の3本の線がねじれ合う密集期は、短期・中期・長期の投資家全員の単価がほぼ同一の価格帯に集中している状態。需給の拮抗期からどちらにブレイクしていくかを観察しましょう。"
        },
        1: {
            "title": "Lesson 2: 出来高（Volume）が物語る『大口のステルス足跡』",
            "desc": "価格は出来高を伴わない『ダマシ』を作ることができますが、出来高そのものは大口の売買事実であるため嘘をつけません。出来高の急減（売り枯れ）は、その価格帯に売り手がいないことを証明し、逆に急増（点火）は新規の大口資金流入を示します。常に出来高を最優先で見てください。"
        },
        2: {
            "title": "Lesson 3: ボラティリティ収縮（スクイーズ）と爆発の物理法則",
            "desc": "ボラティリティは、極限まで収縮（バンド幅が極小）すると、その後必ず上下どちらかに大拡散（ブレイク）に向かうという強い物理法則を持っています。収縮期は仕込みを焦らず、エネルギーが充填されるのを静かに待ち伏せる期間です。"
        },
        3: {
            "title": "Lesson 4: 下降トレンドから上昇トレンドへの大転換期を捉える",
            "desc": "長期のダウントレンドから底固めを経て、移動平均線が集まり始める時期は『最もローリスクな待ち伏せ局面』です。下降を続けていた200日移動平均線を価格が上抜ける瞬間は、大口の『トレンド転換の意志』が現れる最大の節目となります。"
        },
        4: {
            "title": "Lesson 5: 誰も売り急がない『極限の売り枯れ（需給の真空）』の妙味",
            "desc": "出来高が20日平均の0.5倍以下に細る現象は、需給が完全に膠着し『市場参加者の誰もがもう売り急いでいない状態』を示します。売り手が消え失せた板は、わずかな買いが入るだけで上空へ弾き飛ぶ『需給の真空』を形成します。"
        },
        5: {
            "title": "Lesson 6: 上昇第一波の後に必ず訪れる絶望の『初押し（ふるい落とし）』",
            "desc": "大相場の初動（State 4）が始まった後、大衆を振り落とすために一時的に出来高を削って急落させる『ふるい落とし（Shakeout）』が発生します。ここをキープし、出来高が完全に売り枯れる最後の押し目こそが、本上昇へ移行する前の仕込みの黄金位置となります。"
        }
    }
    
    lesson_no = yday % len(curriculum_cycle)
    return curriculum_cycle.get(lesson_no, curriculum_cycle[0])


def generate_review_corner() -> str:
    """
    ⑥：復習コーナー (Review Corner)
    `early_history.csv` の台帳データを動的にロードし、
    登録時価格と本日の現在値を最新株価キャッシュから突き合わせて「答え合わせ」を実行。
    """
    history_file = Path("early_history.csv")
    prefix = "## 🔁 【復習コーナー（Review Corner）】\n"
    if not history_file.exists():
        return prefix + "  ・過去の授業データが蓄積され次第、明日から復習コーナーが開講されます。\n"
    try:
        df = pd.read_csv(history_file)
        if df.empty:
            return prefix + "  ・過去の授業データが蓄積され次第、明日から復習コーナーが開講されます。\n"
            
        unique_dates = sorted(df["date"].unique())
        if len(unique_dates) < 1:
            return prefix + "  ・復習用の蓄積データがまだありません。今後のスキャンによって自動構築されます。\n"
            
        prev_date = unique_dates[-1]  # 最新の過去登録日
        yesterday_items = df[df["date"] == prev_date].head(2)  # 最大2件をピックアップ
        
        review_text = prefix
        review_text += f"直近の授業（ {prev_date} 検出分）に登場した教材たちの『その後の需給変化』を動的に答え合わせ・復習します。\n\n"
        
        for idx, r in yesterday_items.iterrows():
            ticker = r["ticker"]
            price_path = PRICES_DIR / f"{ticker}.csv"
            if not price_path.exists():
                continue
            df_price = pd.read_csv(price_path, index_col=0)
            df_price.index = pd.to_datetime(df_price.index, errors="coerce")
            df_price = df_price.dropna(how="all").sort_index()
            
            orig_close = float(r["close"])
            curr_close = float(df_price["Close"].iloc[-1])
            curr_vol_ratio = float(df_price["Volume"].iloc[-1] / df_price["Volume"].rolling(20).mean().iloc[-1])
            
            perf = (curr_close - orig_close) / orig_close * 100
            status_icon = "📈" if perf >= 0 else "📉"
            
            review_text += f"### {status_icon} 【答え合わせ】 {r['name']} ({ticker})\n"
            review_text += f"  ・登録時価格: {orig_close:.1f}円 ➔ 本日終値: {curr_close:.1f}円 (累積騰落: {perf:+.1f}%)\n"
            review_text += f"  ・本日出来高: {curr_vol_ratio:.2f}倍 (登録時出来高: {r['vol_ratio']:.2f}倍)\n"
            
            # 教育的講評の自動出し分け
            if curr_vol_ratio >= 1.8:
                review_text += f"  ・💡【AI先生の講評】: 出来高が【 {curr_vol_ratio:.2f}倍 】と急増しました。昨日解説した『大口の点火の狼煙』がまさに本日発生した瞬間です。TradingViewでブレイクの推移を今すぐ復習してください。\n\n"
            elif perf >= 1.5:
                review_text += f"  ・💡【AI先生の講評】: 出来高は穏やかなまま、価格がしっかりと反発を始めています。需給の買い優位への傾きが順調に進んでいるサインです。支持線の強さを確認してください。\n\n"
            else:
                review_text += f"  ・💡【AI先生の講評】: 出来高は依然として低く、理想的な『売り枯れもみ合い』を継続してエネルギーを熟成させています。次の拡散へのカウントダウンをじっくり見守りましょう。\n\n"
        return review_text
    except Exception as e:
        return prefix + f"  ・復習データの自動算出中にエラーが発生しました: {e}\n"


def notify_early_watch(candidates: list[dict], date_str: str, evolution_alerts: list[str] = None):
    if not candidates and not evolution_alerts:
        print("本日のEarly（移動平均大収縮）予備軍は0件です。通知をスキップします。")
        return

    if not (GMAIL_USER and GMAIL_PASS and NOTIFICATION_EMAIL):
        print("警告: メールの認証情報、または通知先アドレスが未設定です。")
        return

    from state5_explainable_engine import State5ExplainableEngine

    msg = MIMEMultipart()
    msg["From"] = f"{SENDER_NAME} <{GMAIL_USER}>"
    msg["To"] = NOTIFICATION_EMAIL
    
    if evolution_alerts:
        msg["Subject"] = f"【AI相場学校 8.5】{date_str} 予備軍の『進化』を検知！ (他 {len(candidates)} 銘柄)"
    else:
        msg["Subject"] = f"【AI相場学校 8.5】{date_str} 本日の『Today's Lesson』開講のお知らせ"

    body = ""
    
    # --- ①：Today's Lesson (カリキュラム体系化) をメールの最上部に掲出 ---
    lesson = get_curriculum_lesson(date_str)
    body += "## ━━━━━━━━━━━━━━━━━━\n"
    body += f"## 🧑‍🏫 【Today's Lesson】本日の学習カリキュラム\n"
    body += f"## 📌 『 {lesson['title']} 』\n"
    body += "## ━━━━━━━━━━━━━━━━━━\n"
    body += f"{lesson['desc']}\n"
    body += "## ━━━━━━━━━━━━━━━━━━\n\n"

    # --- ⑤：今日の最優秀教材 (Best Learning Chart) を特選掲出 ---
    best_chart = State5ExplainableEngine.select_best_learning_chart(candidates)
    if best_chart:
        body += "## ━━━━━━━━━━━━━━━━━━\n"
        body += best_chart
        body += "## ━━━━━━━━━━━━━━━━━━\n\n"

    if evolution_alerts:
        body += "## ━━━━━━━━━━━━━━━━━━\n"
        body += "## 🔔 【本日の予備軍・自律成長（進化）アラート】\n"
        body += "## ━━━━━━━━━━━━━━━━━━\n"
        for alert in evolution_alerts:
            body += f"  {alert}\n"
        body += "## ━━━━━━━━━━━━━━━━━━\n\n"

    body += f"# 💡 【Early Watch 8.5】{date_str} トレンド転換初期・本日の教材銘柄リスト\n"
    body += "※買い推奨ツールではありません。移動平均線の収縮・拡散の力学を、毎日チャートを開いて学ぶための「究極の教材リスト」です。\n"
    body += "----------------------------------------\n\n"

    for idx, c in enumerate(candidates, 1):
        links = State5ExplainableEngine.get_chart_links(c["ticker"])["all_markdown"]
        body += f"## {idx}. {c['name']} ({c['ticker']}) {links}\n"
        body += f"### 🎓 【教材難易度】: **{c['difficulty']}**\n"
        body += f"### 📊 【トレンド成熟度】: **{c['trend_stage']}**\n"
        body += f"### 🚀 【拡散準備度 (Expansion Readiness)】: **【 {c['readiness']} 】**\n"
        
        body += f"### ⚡ 【エネルギー蓄積度 (Compression Score)】: **【 {c['compression_score']} 点 】 (100点満点)**\n"
        body += f"      ・昨日比  : {c['chg_score_1d']:+d} 点 ｜ 1週間前比: {c['chg_score_1w']:+d} 点\n"
        body += f"      ・RSI(14) : {c['rsi14']:.1f}% (昨日比: {c['chg_rsi_1d']:+.1f}% ｜ 1週間前比: {c['chg_rsi_1w']:+.1f}%)\n"
        body += f"      ・出来高  : {c['vol_ratio']:.2f}倍 (昨日比: {c['chg_vol_1d']:+.2f} ｜ 1週間前比: {c['chg_vol_1w']:+.2f})\n"
        body += f"      ・BB幅    : {c['bb_width']:.1f}% (昨日比: {c['chg_bb_1d']:+.1f}% ｜ 1週間前比: {c['chg_bb_1w']:+.1f}%)\n\n"
        
        # ②：動的チャート観察チェックリストの印字
        body += "🔍 【本日のチャート観察チェックリスト】\n"
        for p in c["checklist"]:
            body += f"  {p}\n"
        body += "\n"
        
        # ③：明日見るポイントの印字
        body += f"🎯 {c['tomorrow_mission']}\n\n"
        
        body += f"{c['similar_winners_desc']}\n\n"
        
        body += "【Moving Average Trend Score (傾き判定)】\n"
        body += f"  ・25日移動平均線 : 【 {c['s25']} 】  (短期トレンドの方向)\n"
        body += f"  ・75日移動平均線 : 【 {c['s75']} 】  (中期トレンドの方向)\n"
        body += f"  ・200日移動平均線: 【 {c['s200']} 】  (長期トレンド of 方向)\n\n"
        
        body += "【基本テクニカル】\n"
        body += f"  ・MA密集度  : **{c['congestion_width']:.2f}%** (基準: 5%以下 / 密集継続: {c['congestion_duration']}営業日)\n"
        body += f"  ・52週高値比: {c['dist_52w']:+.1f}%\n\n"
        
        body += f"📢 **【今回の動的着眼点】: {c['edu_comment']}**\n"
        body += "----------------------------------------\n\n"

    # --- ⑥：動的復習コーナー (Review Corner) の差し込み ---
    body += "\n"
    body += generate_review_corner()
    body += "\n"

    # --- ④：AI相場先生の定点解説＆宿題の差し込み ---
    body += "\n"
    body += State5ExplainableEngine.generate_human_learning_summary(candidates)
    body += "\n"
    
    # --- ⑧：AI Research Note（検証仮説）の差し込み ---
    body += State5ExplainableEngine.generate_ai_research_note()
    body += "\n"
    
    # --- ⑨：実践フェーズへの橋渡し（Practice）の差し込み ---
    body += "## 🏃 【本日の相場実践トレーニング (Practice)】\n"
    body += "  ・本日は、教材銘柄リストの中から『あなたが最も気になった1銘柄』だけを選び、TradingViewでチャートを【10分間】じっくり観察してください。\n"
    body += "  ・エントリーは不要です。チェックリストが明日どう変化するか、ローソク足の『形』と出来高の『呼吸』だけを無心で眺めてみてください。この毎日の蓄積が、あなたの需給眼を別次元へと引き上げます。\n\n"
    
    body += "※本メールは、チャートの『呼吸（収縮と拡散）』や移動平均線の需給力学を学び、相場感を養うための研究用レポートです。投資の勉強材料としてTradingViewのリンクから実際の形を確認し、イメージを膨らませてください。\n"

    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASS)
        server.send_message(msg)
    print("毎朝のEarly（予備軍・学習特化型）メールを正常に送信しました。")


def main():
    try:
        if not UNIVERSE_CSV.exists():
            print(f"宇宙ファイル {UNIVERSE_CSV} が存在しません。")
            return

        df_uni = pd.read_csv(UNIVERSE_CSV)
        tickers = df_uni["ticker"].dropna().tolist()
        tickers = [normalize_ticker(t) for t in tickers]
        
        name_map = dict(zip(df_uni["ticker"].map(normalize_ticker), df_uni["name"]))

        candidates = []
        priority_candidates = []
        latest_date = None

        print(f"=== Early Watch 予備軍スキャンの稼働を開始します (対象: {len(tickers)} 銘柄) ===")

        from state5_explainable_engine import State5ExplainableEngine

        error_count = 0
        debug_print_count = 0

        for idx, t in enumerate(tickers):
            price_path = PRICES_DIR / f"{t}.csv"
            if not price_path.exists():
                continue

            try:
                df_raw = pd.read_csv(price_path, index_col=0)
                df_raw.index = pd.to_datetime(df_raw.index, errors="coerce")
                df_raw = df_raw.dropna(how="all").sort_index()
                
                if len(df_raw) < 155:
                    continue

                d = EarlyStateEngine.calculate_early_indicators(df_raw)
                
                row = d.iloc[-1]
                row_1d = d.iloc[-2]
                row_1w = d.iloc[-6]
                
                if latest_date is None:
                    latest_date = d.index[-1].strftime("%Y-%m-%d")

                if row["turnover_avg20_million"] < TH_MIN_TURNOVER:
                    continue

                if debug_print_count < 10:
                    print(f"  [デバッグ] 銘柄 {t} ➔ データ行数: {len(df_raw)} / 20日平均売買代金: {row['turnover_avg20_million']:.2f} 百万円 (しきい値: {TH_MIN_TURNOVER:.2f} 百万円)")
                    debug_print_count += 1

                # --- 【テスト用】：条件を if True: にして強制的に全件抽出 ---
                if True:
                    s25, s75, s200 = EducationalAnalyzer.get_ma_slope_symbols(row)
                    trend_stage = EducationalAnalyzer.get_trend_stage(row, s25, s75, s200)
                    readiness = EducationalAnalyzer.get_expansion_readiness(row, s25, s75, s200)
                    
                    chart_pattern = "緩やかな上昇トレンド"
                    if hasattr(State5ExplainableEngine, "get_chart_pattern"):
                        chart_pattern = State5ExplainableEngine.get_chart_pattern(df_raw)
                    elif hasattr(State5ExplainableEngine, "detect_chart_pattern"):
                        chart_pattern = State5ExplainableEngine.detect_chart_pattern(df_raw)
                        
                    # ②：完全動的な個別教育着眼点コメント
                    edu_comment = EducationalAnalyzer.generate_educational_comment(row, trend_stage, chart_pattern)
                    
                    # ②：チャート観察チェックリスト（今日見る3つのポイント）
                    checklist = EducationalAnalyzer.generate_chart_checklist(row)
                    
                    # ③：明日見るポイントの動的生成
                    tomorrow_mission = EducationalAnalyzer.generate_tomorrow_mission(row)
                    
                    # ⑦：難易度レベルの動的判定
                    difficulty = State5ExplainableEngine.get_difficulty_level(row)
                    
                    # Change Detection (昨日比、1週間前比の差分Δの算出)
                    comp_today = int(row["compression_score"])
                    chg_score_1d = comp_today - int(row_1d["compression_score"])
                    chg_score_1w = comp_today - int(row_1w["compression_score"])
                    
                    rsi_today = float(row["rsi14"])
                    chg_rsi_1d = rsi_today - float(row_1d["rsi14"])
                    chg_rsi_1w = rsi_today - float(row_1w["rsi14"])
                    
                    vol_today = float(row["vol_ratio_20"])
                    chg_vol_1d = vol_today - float(row_1d["vol_ratio_20"])
                    chg_vol_1w = vol_today - float(row_1w["vol_ratio_20"])
                    
                    bb_today = float(row["bb_width"])
                    chg_bb_1d = bb_today - float(row_1d["bb_width"])
                    chg_bb_1w = bb_today - float(row_1w["bb_width"])

                    if hasattr(State5ExplainableEngine, "get_type0_matching_rate"):
                        type0_match = State5ExplainableEngine.get_type0_matching_rate(row)
                    else:
                        type0_match = 0
                        
                    if hasattr(State5ExplainableEngine, "get_similar_historical_winners"):
                        similar_winners_desc = State5ExplainableEngine.get_similar_historical_winners(row, type0_match)
                    else:
                        similar_winners_desc = "  ・【過去類似チャート】: [逆引きエンジン調整中]"
                    
                    candidates.append({
                        "ticker": t,
                        "name": name_map.get(t, t),
                        "congestion_width": row["ma_congestion_width_pct"],
                        "congestion_duration": int(row["congestion_duration"]) if not pd.isna(row["congestion_duration"]) else 0,
                        "rsi14": rsi_today,
                        "bb_width": bb_today,
                        "vol_ratio": vol_today,
                        "dist_to_52w_high": row["dist_to_52w_high"],
                        "dist_52w": row["dist_to_52w_high"],
                        "ma75": row["ma75"],
                        "s25": s25, "s75": s75, "s200": s200,
                        "trend_stage": trend_stage,
                        "compression_score": comp_today,
                        "readiness": readiness,
                        "edu_comment": edu_comment,
                        "similar_winners_desc": similar_winners_desc,
                        "checklist": checklist,                # 追加
                        "tomorrow_mission": tomorrow_mission,    # 追加
                        "difficulty": difficulty,                # 追加
                        "close": row["Close"],                  # 復習用に追加
                        
                        # Change Detection
                        "chg_score_1d": chg_score_1d,
                        "chg_score_1w": chg_score_1w,
                        "chg_rsi_1d": rsi_today - float(row_1d["rsi14"]),
                        "chg_rsi_1w": rsi_today - float(row_1w["rsi14"]),
                        "chg_vol_1d": vol_today - float(row_1d["vol_ratio_20"]),
                        "chg_vol_1w": vol_today - float(row_1w["vol_ratio_20"]),
                        "chg_bb_1d": bb_today - float(row_1d["bb_width"]),
                        "chg_bb_1w": bb_today - float(row_1w["bb_width"])
                    })
            except Exception as e:
                import traceback
                if error_count < 3:
                    error_count += 1
                    print(f"\n[デバッグ警告] 銘柄 {t} の処理中に例外（エラー）を検知しました:")
                    print(f"エラー内容: {e}")
                    print("--- スタックトレース（発生場所） ---")
                    print(traceback.format_exc())
                    print("-----------------------------------\n")
                continue

        if candidates:
            sorted_candidates = sorted(candidates, key=lambda x: x["congestion_width"])
            priority_candidates = sorted_candidates[:5]

        # ==========================================
        # ★【Version 7.22 新設】：過去の予備軍の自動成績追跡・進化判定を起動 ★
        # ==========================================
        evolution_alerts = []
        try:
            print("\n=== Version 7.22: 予備軍（Early Watch）の自動成績追跡・進化判定を起動します ===")
            
            from early_history_logger import EarlyHistoryLogger
            EarlyHistoryLogger.log_early_candidates(priority_candidates, latest_date, config)
            
            from early_performance_tracker import EarlyPerformanceTracker
            evolution_alerts = EarlyPerformanceTracker.track_and_detect_evolutions(config)
            
            print(f"=== Version 7.22: 予備軍追跡・進化判定が正常完了しました (本日発生のアラート: {len(evolution_alerts)} 件) ===")
            
        except Exception as e:
            print(f"【エラーログ】Version 7.22 予備軍追跡中に例外が発生しました: {e}")

        # 毎朝のEarlyメール送信
        notify_early_watch(priority_candidates, latest_date, evolution_alerts)

    except Exception as e:
        print(f"【エラーログ】Early Watch 監視中に例外が発生しました: {e}")


if __name__ == "__main__":
    main()
