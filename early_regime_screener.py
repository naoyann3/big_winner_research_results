# early_regime_screener.py (Version 8.4 - Learning OS with Today's Lesson)
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
SENDER_NAME = "Sniper OS - Early Watch 8.4"

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
        
        # 1つ前の傾き
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
        固定テンプレートを廃止し、ローソク足と移動平均線（25MA, 75MA, 200MA）の位置関係を
        マルチプロファイリングして、銘柄ごとに全く異なる具体的な着眼点を出力します。
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
        
        # 乖離率
        dev75 = abs(close - ma75) / ma75 * 100 if ma75 > 0 else 0.0

        # パターン1: 長期抵抗突破検証（200日線が下向き、価格が200日線付近または上抜け）
        if m200_slope < -0.15 and (close >= ma200 or abs(close - ma200)/ma200 <= 0.02):
            return (
                f"【今回の動的着眼点】: 長期的に下降トレンドの蓋として機能していた『長期200日線（{ma200:.1f}円）』をブレイクできるかどうかの極めて重要な転換点です。 "
                f"200日線に頭を抑えられて押し返されるダマシになるか、それとも出来高を伴って本上抜けへ昇華するか、大口の介入意志を注視してください。"
            )
            
        # パターン2: 75日線支持（サポート）検証
        elif dev75 <= 1.2 and ma75 >= ma200:
            return (
                f"【今回の動的着眼点】: 上向きの中期トレンドラインである『75日移動平均線（{ma75:.1f}円）』に対して、株価がわずか {dev75:.2f}% と完全近接しています。 "
                f"ここが絶好の押し目（下値支持帯）として機能し、出来高の極限減少（売り枯れ）を経て、上方向への反発動意（買い手の出現）が起きるかを観察するのに最高の教材です。"
            )
            
        # パターン3: エネルギー極限充填ボックスブレイク前夜
        elif width <= 1.5 and duration >= 15:
            return (
                f"【今回の動的着眼点】: 3本の移動平均線がわずか {width:.2f}% という極限レベルで密集し、その膠着が 【 {duration}営業日 】 も継続しています。 "
                f"煮詰まりは極限に達しており、直近のボックス上限をブレイクする際に、出来高が2〜3倍以上に『再点火』するかどうかを毎日観察してください。"
            )
            
        # パターン4: ボックスもみ合い煮詰まり
        elif "ボックス" in pattern and duration >= 15:
            return (
                f"【今回の動的着眼点】: 『{pattern}』というレンジ内で、売りと買いのバランスが均衡しています。 "
                f"出来高比率が {vol_ratio:.2f}倍 まで著しく細っているのは、投げ売る個人が全員いなくなった『売り枯れの呼吸』です。この極限状態からどちらに拡散エネルギーが放たれるかを定点観測してください。"
            )
            
        # パターン5: 25日線上向き・新鮮なトレンド
        elif m25_slope > 0.3 and close > ma25:
            return (
                f"【今回の動的着眼点】: 短期25日移動平均線が【上向き】へ反転し、ローソク足がその上に乗る新鮮な上昇動意の形です。 "
                f"ここから上放れが本格化する際、上方に存在する長期200日線（抵抗帯）とぶつかった時の押し引きを先行的に脳内シミュレーションしておきましょう。"
            )
            
        # パターン6: デフォルト（底打ち・ねじれ期）
        else:
            return (
                f"【今回の動的着眼点】: 移動平均線の密集度が極めて狭い、典型的な『エネルギーの限界充填期』です。 "
                f"下値支持帯である75日移動平均線付近で株価がしっかりと下げ止まり、出来高が消滅する需給の真空状態をじっくり観察してください。"
            )


def get_todays_lesson_theme(candidates: list[dict]) -> dict:
    """
    ④：Today's Lesson（授業テーマ）を本日の候補銘柄の分布から自動的にクラスター分類して提示
    """
    if not candidates:
        return {
            "title": "休講：市場の需給調整期における『静観の技術』を学ぶ",
            "desc": "本日は移動平均大収縮の合格者がいません。プロが最も大切にする『無理にポジションを取らず、資金を温存する（待つのも相場）』という、最も重要な自己防衛技術を身につけるための絶好の1日です。"
        }
        
    avg_bb = np.mean([c["bb_width"] for c in candidates])
    avg_vol = np.mean([c["vol_ratio"] for c in candidates])
    avg_congestion = np.mean([c["congestion_width"] for c in candidates])
    
    # 1. 密集度が異常に狭い場合
    if avg_congestion <= 0.15:
        return {
            "title": "第1講：3本の移動平均線が一本のロープのように絡み合う『エネルギー極限密集』を学ぶ",
            "desc": f"本日の候補は3MA密集度の平均がなんと【 {avg_congestion:.2f}% 】という、チャート上でほぼ線が一本に重なって見えるほどの超密集状態を起こしています。これは大口のステルス仕込みが極限まで進んだ痕跡であり、ここから始まる『拡散（ブレイク）』の爆発力と予兆を掴む絶好の1日です。"
        }
    # 2. ボラティリティが極限収縮している場合
    elif avg_bb <= 5.0:
        return {
            "title": "第2講：ボラティリティが死んだ超収縮期（極限スクイーズ）の力学を学ぶ",
            "desc": f"本日の教材は平均BB幅が【 {avg_bb:.1f}% 】と、歴史的に見ても驚異的なボラティリティの押し殺しを示しています。ボラティリティは『収縮の極限に達すると、次は必ず上下どちらかに大爆発（拡散）に向かう』という普遍的な物理法則を持っています。嵐の前の、この静寂を観察してください。"
        }
    # 3. 出来高が極限まで枯れている場合
    elif avg_vol <= 0.50:
        return {
            "title": "第3講：投げ売り圧力が完全に消滅した『売り枯れの呼吸（需給の真空）』を観察する",
            "desc": f"本日の候補は平均出来高比率が20日平均の【 {avg_vol:.2f}倍 】と極限まで激減しています。市場参加者の中に売り急ぐ人が一人もいなくなった『売り枯れの呼吸』の状態です。この需給の空白地帯から、大口がわずかな買いを入れた瞬間に株価が上空へ弾き飛ばされる『再点火』の物理法則を追跡しましょう。"
        }
    # 4. デフォルトテーマ
    else:
        return {
            "title": "第4講：下降トレンドから底固めを経て始まる『トレンド転換の最も新鮮な初動』を学ぶ",
            "desc": "本日は、長期的な下降もみ合いから、移動平均線が集まり始めた『需給の転換初期（目覚めの瞬間）』の銘柄が集まっています。短期（25日）が中期（75日）をねじりながら上抜ける際の、日々の日足の押し引きを定点観測する練習をしましょう。"
        }


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
        msg["Subject"] = f"【Early Watch 8.4】{date_str} 予備軍の『進化』を検知！ (他 {len(candidates)} 銘柄)"
    else:
        msg["Subject"] = f"【Early Watch 8.4】{date_str} 本日の『Today's Lesson』開講のお知らせ"

    body = ""
    
    # --- ④：Today's Lesson (授業テーマ) をメールの最上部に掲出 ---
    lesson = get_todays_lesson_theme(candidates)
    body += "## ━━━━━━━━━━━━━━━━━━\n"
    body += f"## 🧑‍🏫 【Today's Lesson】本日の学習テーマ\n"
    body += f"## 📌 『 {lesson['title']} 』\n"
    body += "## ━━━━━━━━━━━━━━━━━━\n"
    body += f"{lesson['desc']}\n"
    body += "## ━━━━━━━━━━━━━━━━━━\n\n"

    if evolution_alerts:
        body += "## ━━━━━━━━━━━━━━━━━━\n"
        body += "## 🔔 【本日の予備軍・自律成長（進化）アラート】\n"
        body += "## ━━━━━━━━━━━━━━━━━━\n"
        body += "過去にEarly Watchで検出され、追跡中の銘柄に、本日劇的な変化が起きました：\n\n"
        for alert in evolution_alerts:
            body += f"  {alert}\n"
        body += "## ━━━━━━━━━━━━━━━━━━\n\n"

    body += f"# 💡 【Early Watch 8.4】{date_str} トレンド転換初期・本日の教材銘柄リスト\n"
    body += "※買い推奨ツールではありません。移動平均線の収縮・拡散の力学を、毎日チャートを開いて学ぶための「究極の教材リスト」です。\n"
    body += "----------------------------------------\n\n"

    for idx, c in enumerate(candidates, 1):
        links = State5ExplainableEngine.get_chart_links(c["ticker"])["all_markdown"]
        body += f"## {idx}. {c['name']} ({c['ticker']}) {links}\n"
        body += f"### 📊 【トレンド成熟度】: **{c['trend_stage']}**\n"
        body += f"### 🚀 【拡散準備度 (Expansion Readiness)】: **【 {c['readiness']} 】**\n"
        
        body += f"### ⚡ 【エネルギー蓄積度 (Compression Score)】: **【 {c['compression_score']} 点 】 (100点満点)**\n"
        body += f"      ・昨日比  : {c['chg_score_1d']:+d} 点 ｜ 1週間前比: {c['chg_score_1w']:+d} 点\n"
        body += f"      ・RSI(14) : {c['rsi14']:.1f}% (昨日比: {c['chg_rsi_1d']:+.1f}% ｜ 1週間前比: {c['chg_rsi_1w']:+.1f}%)\n"
        body += f"      ・出来高  : {c['vol_ratio']:.2f}倍 (昨日比: {c['chg_vol_1d']:+.2f} ｜ 1週間前比: {c['chg_vol_1w']:+.2f})\n"
        body += f"      ・BB幅    : {c['bb_width']:.1f}% (昨日比: {c['chg_bb_1d']:+.1f}% ｜ 1週間前比: {c['chg_bb_1w']:+.1f}%)\n\n"
        
        body += f"{c['similar_winners_desc']}\n\n"
        
        body += "【Moving Average Trend Score (傾き判定)】\n"
        body += f"  ・25日移動平均線 : 【 {c['s25']} 】  (短期トレンドの方向)\n"
        body += f"  ・75日移動平均線 : 【 {c['s75']} 】  (中期トレンドの方向)\n"
        body += f"  ・200日移動平均線: 【 {c['s200']} 】  (長期トレンド of 方向)\n\n"
        
        body += "【基本テクニカル】\n"
        body += f"  ・MA密集度  : **{c['congestion_width']:.2f}%** (基準: 5%以下 / 密集継続: {c['congestion_duration']}営業日)\n"
        body += f"  ・52週高値比: {c['dist_52w']:+.1f}%\n\n"
        
        body += f"📢 **{c['edu_comment']}**\n"
        body += "----------------------------------------\n\n"

    # --- ⑦: Human Learning Comment & ⑥: AI Research Note の結合差し込み ---
    body += "\n"
    body += State5ExplainableEngine.generate_human_learning_summary(candidates)
    body += "\n"
    body += State5ExplainableEngine.generate_ai_research_note()
    body += "\n"
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
                # 日付インデックスの強制クレンジング
                df_raw = pd.read_csv(price_path, index_col=0)
                df_raw.index = pd.to_datetime(df_raw.index, errors="coerce")
                df_raw = df_raw.dropna(how="all").sort_index()
                
                # 昨日比・一週間前比を計算するため、最低155行必要
                if len(df_raw) < 155:
                    continue

                d = EarlyStateEngine.calculate_early_indicators(df_raw)
                
                # 本日（最新行）、昨日（1行前）、一週間前（5営業日前）のデータを抽出
                row = d.iloc[-1]
                row_1d = d.iloc[-2]
                row_1w = d.iloc[-6]
                
                if latest_date is None:
                    latest_date = d.index[-1].strftime("%Y-%m-%d")

                # 最低流動性（売買代金）チェック (1,000万円以上)
                if row["turnover_avg20_million"] < TH_MIN_TURNOVER:
                    continue

                # 最初の10件のみ、進行確認のためにログをプリント
                if debug_print_count < 10:
                    print(f"  [デバッグ] 銘柄 {t} ➔ データ行数: {len(df_raw)} / 20日平均売買代金: {row['turnover_avg20_million']:.2f} 百万円 (しきい値: {TH_MIN_TURNOVER:.2f} 百万円)")
                    debug_print_count += 1

                # --- 【テスト用】：条件を if True: にして強制的に全件抽出 ---
                if True:
                    s25, s75, s200 = EducationalAnalyzer.get_ma_slope_symbols(row)
                    trend_stage = EducationalAnalyzer.get_trend_stage(row, s25, s75, s200)
                    readiness = EducationalAnalyzer.get_expansion_readiness(row, s25, s75, s200)
                    
                    # --- ②：多重防壁仕様のチャートパターン動的判定 ➔ 動的コメントへの連動 ---
                    chart_pattern = "緩やかな上昇トレンド"
                    if hasattr(State5ExplainableEngine, "get_chart_pattern"):
                        chart_pattern = State5ExplainableEngine.get_chart_pattern(df_raw)
                    elif hasattr(State5ExplainableEngine, "detect_chart_pattern"):
                        chart_pattern = State5ExplainableEngine.detect_chart_pattern(df_raw)
                        
                    # 完全動的な教育着眼点コメントの生成
                    edu_comment = EducationalAnalyzer.generate_educational_comment(row, trend_stage, chart_pattern)
                    
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

                    # 【Version 7.23新設】：大化けデータベースから自動逆引き (安全なフォールバック付き)
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
