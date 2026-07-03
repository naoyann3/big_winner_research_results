# early_regime_screener.py (Version 7.2 - Educational Quant)
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
SENDER_NAME = "Sniper OS - Early Watch"

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
        
        # 移動平均の傾き（直近5営業日の変化率 %）
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

        return d


class EducationalAnalyzer:
    """
    ①〜⑤：Version 7.2新設の「学習価値を高めるための評価パラメータ」算出クラス
    """
    @staticmethod
    def get_ma_slope_symbols(latest_row: pd.Series) -> tuple[str, str, str]:
        """
        ①：Moving Average Trend Score (MAの傾き・矢印判定)
        """
        def get_symbol(slope):
            if slope > 0.1: return "↑"
            elif slope < -0.1: return "↓"
            else: return "→"
            
        s25 = get_symbol(latest_row["ma25_slope"])
        s75 = get_symbol(latest_row["ma75_slope"])
        s200 = get_symbol(latest_row["ma200_slope"])
        return s25, s75, s200

    @staticmethod
    def get_trend_stage(latest_row: pd.Series, s25: str, s75: str, s200: str) -> str:
        """
        ②：トレンド成熟度の総合分類
        """
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
    def calculate_compression_score(latest_row: pd.Series) -> int:
        """
        ③：Compression Score (エネルギー蓄積度の100点満点評価)
        """
        score = 0
        width = latest_row["ma_congestion_width_pct"]
        duration = int(latest_row["congestion_duration"])
        bb_width = latest_row["bb_width"]
        
        # 1. 密集度 (最大40点)
        if width <= 1.5: score += 40
        elif width <= 3.0: score += 30
        elif width <= 5.0: score += 20
        else: score += 10
        
        # 2. 密集継続日数 (最大30点)
        if duration >= 25: score += 30
        elif duration >= 15: score += 20
        elif duration >= 5: score += 10
        else: score += 5
        
        # 3. ボラティリティ収縮（BB幅） (最大30点)
        if bb_width <= 5.0: score += 30
        elif bb_width <= 8.0: score += 20
        elif bb_width <= 12.0: score += 10
        else: score += 5
        
        return score

    @staticmethod
    def get_expansion_readiness(latest_row: pd.Series, s25: str, s75: str, s200: str) -> str:
        """
        ④：Expansion Readiness (上方向への拡散準備度：S, A, B, C)
        """
        close = latest_row["Close"]
        ma25 = latest_row["ma25"]
        ma75 = latest_row["ma75"]
        ma200 = latest_row["ma200"]
        
        # Sクラス：25MAが上向き反転、且つ75MA・200MAが下げ止まっており、価格が移動平均の上にある
        if s25 == "↑" and s75 in ["↑", "→"] and close > ma75:
            return "S (極上：本上抜けカウントダウン状態)"
        # Aクラス：25MAは上向き、且つ価格が25日線の上にあるが、長期の200MAがまだ下向き
        elif s25 == "↑" and close > ma25:
            return "A (良好：トレンド発生の兆候あり)"
        # Bクラス：3本移動平均は収縮しているが、25MAがまだ「横ばい(→)」で動き出していない
        elif s25 == "→" and latest_row["ma_congestion_width_pct"] <= 5.0:
            return "B (待機：エネルギー充填中で、反転のきっかけ待ち)"
        # Cクラス：25MAがまだ「下向き(↓)」の、底ばいもみ合い期
        else:
            return "C (未成熟：底固めの途上にあり、点火には数日〜数週間が必要)"

    @staticmethod
    def generate_educational_comment(latest_row: pd.Series, trend_stage: str, pattern: str) -> str:
        """
        ⑤：人間がチャートを見る際の「着眼点」を教えるAI学習コメント
        """
        ma200_slope = latest_row["ma200_slope"]
        days_held = int(latest_row["congestion_duration"])
        
        if ma200_slope < -0.15:
            return (
                f"【今回の学習着眼点】: 200日線がまだ「下向き」の下降トレンド途中にあります。 "
                f"ここからの上放れが、長期的な抵抗（200日線）を上抜けて本物の『トレンド転換（大化けの第一歩）』に発展するか、"
                f"それとも200日線に頭を抑えられて再び下降継続（×）になってしまうかが最大の見どころです。ダマシの動きに注目してください。"
            )
        elif "ボックス" in pattern and days_held >= 20:
            return (
                f"【今回の学習着眼点】: ボックス圏のレンジ内で、エネルギーが 【 {days_held}日営業日 】 にわたって極限まで固められています。 "
                f"直近の上限ラインを陽線で突き抜ける際に、出来高が2〜3倍以上に『再点火』するかどうかを毎日観察してください。出来高を伴わない上抜けはダマシになる良い例です。"
            )
        elif "三角" in pattern or "ウェッジ" in pattern:
            return (
                f"【今回の学習着眼点】: チャート形状が『{pattern}』という、煮詰まりの最終形状を迎えています。 "
                f"収縮の極限（三角の先端）に近づくにつれて、本当に出来高が限界まで枯渇し、その後にどの方向にエネルギーが『拡散』していくかを毎日定点観測するのに最適な教材です。"
            )
        else:
            return (
                f"【今回の学習着眼点】: 移動平均線の密集度が極めて狭い、典型的な『エネルギーの限界充填期』です。 "
                f"下値支持帯である75日移動平均線（ピンク色）で株価がしっかりと下げ止まり、出来高が消滅する『売り枯れの呼吸』をじっくり観察してください。"
            )


def notify_early_watch(candidates: list[dict], date_str: str):
    """
    【Version 7.2大刷新版】：
    Moving Average Trend Score、エネルギー蓄積度、拡散準備度、
    および「AIからの学習着眼点コメント」を完全搭載した、極上のクオンツ学習教材メール。
    """
    if not candidates:
        print("本日のEarly（移動平均大収縮）予備軍は0件です。通知をスキップします。")
        return

    if not (GMAIL_USER and GMAIL_PASS and NOTIFICATION_EMAIL):
        print("警告: メールの認証情報、または通知先アドレスが未設定です。")
        return

    from state5_explainable_engine import State5ExplainableEngine

    msg = MIMEMultipart()
    msg["From"] = f"{SENDER_NAME} <{GMAIL_USER}>"
    msg["To"] = NOTIFICATION_EMAIL
    msg["Subject"] = f"【Early Watch】{date_str} 移動平均大収縮・予備軍 {len(candidates)} 銘柄"

    body = f"# 💡 【Early Watch】{date_str} トレンド転換初期・予備軍リスト\n"
    body += "※買い推奨ツールではありません。移動平均線の収縮・拡散の力学を、毎日チャートを開いて学ぶための「究極の教材リスト」です。\n"
    body += "----------------------------------------\n\n"

    for idx, c in enumerate(candidates, 1):
        links = State5ExplainableEngine.get_chart_links(c["ticker"])["all_markdown"]
        body += f"## {idx}. {c['name']} ({c['ticker']}) {links}\n"
        body += f"### 📊 【トレンド成熟度】: **{c['trend_stage']}**\n"
        body += f"### 🚀 【拡散準備度 (Expansion Readiness)】: **【 {c['readiness']} 】**\n"
        body += f"### ⚡ 【エネルギー蓄積度 (Compression Score)】: **【 {c['compression_score']} 点 】 (100点満点)**\n\n"
        
        # 移動平均線のトレンドスコア
        body += "【Moving Average Trend Score (傾き判定)】\n"
        body += f"  ・25日移動平均線 : 【 {c['s25']} 】  (短期トレンドの方向)\n"
        body += f"  ・75日移動平均線 : 【 {c['s75']} 】  (中期トレンドの方向)\n"
        body += f"  ・200日移動平均線: 【 {c['s200']} 】  (長期トレンドの方向)\n\n"
        
        # テクニカル詳細
        body += "【基本テクニカル】\n"
        body += f"  ・MA密集度  : **{c['congestion_width']:.2f}%** (基準: 5%以下 / 密集継続: {c['congestion_duration']}営業日)\n"
        body += f"  ・RSI(14)   : {c['rsi14']:.1f}% / BB幅: {c['bb_width']:.1f}% / 出来高比率: {c['vol_ratio']:.2f}倍\n"
        body += f"  ・52週高値比: {c['dist_52w']:+.1f}%\n\n"
        
        # ⑤ AIによる「今回の学習ポイント」
        body += f"📢 **{c['edu_comment']}**\n"
        body += "----------------------------------------\n\n"

    body += "\n※本メールは、チャートの『呼吸（収縮と拡散）』や移動平均線の需給力学を学び、相場感を養うための研究用レポートです。投資の勉強材料としてTradingViewのリンクから実際の形を確認し、イメージを膨らませてください。\n"

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

        for idx, t in enumerate(tickers):
            price_path = PRICES_DIR / f"{t}.csv"
            if not price_path.exists():
                continue

            try:
                # 日付インデックスの強制クレンジング
                df_raw = pd.read_csv(price_path, index_col=0)
                df_raw.index = pd.to_datetime(df_raw.index, errors="coerce")
                df_raw = df_raw.dropna(how="all").sort_index()
                
                if len(df_raw) < 150:
                    continue

                d = EarlyStateEngine.calculate_early_indicators(df_raw)
                row = d.iloc[-1]
                
                if latest_date is None:
                    latest_date = d.index[-1].strftime("%Y-%m-%d")

                # 最低流動性（売買代金）チェック (1,000万円以上)
                if row["turnover_avg20_million"] < TH_MIN_TURNOVER:
                    continue

                # --- 【テスト用】：条件を if True: にして強制的に全件抽出 ---
                if True:
                    # ① MAの傾き（矢印）判定
                    s25, s75, s200 = EducationalAnalyzer.get_ma_slope_symbols(row)
                    
                    # ② トレンド成熟度の総合判定
                    trend_stage = EducationalAnalyzer.get_trend_stage(row, s25, s75, s200)
                    
                    # ③ Compression Score (エネルギー蓄積度 100点満点)
                    comp_score = EducationalAnalyzer.calculate_compression_score(row)
                    
                    # ④ Expansion Readiness (拡散準備度ランク)
                    readiness = EducationalAnalyzer.get_expansion_readiness(row, s25, s75, s200)
                    
                    # 簡易的なチャートパターン判定
                    chart_pattern = State5ExplainableEngine.get_chart_pattern(df_raw)
                    
                    # ⑤ AIによる学習着眼点コラムの自動生成
                    edu_comment = EducationalAnalyzer.generate_educational_comment(row, trend_stage, chart_pattern)
                    
                    candidates.append({
                        "ticker": t,
                        "name": name_map.get(t, t),
                        "congestion_width": row["ma_congestion_width_pct"],
                        "congestion_duration": int(row["congestion_duration"]),
                        "rsi14": row["rsi14"],
                        "bb_width": row["bb_width"],
                        "vol_ratio": row["vol_ratio_20"],
                        "dist_to_52w_high": row["dist_to_52w_high"],
                        "dist_52w": row["dist_to_52w_high"],
                        "ma75": row["ma75"],
                        # 学習用新指標
                        "s25": s25, "s75": s75, "s200": s200,
                        "trend_stage": trend_stage,
                        "compression_score": comp_score,
                        "readiness": readiness,
                        "edu_comment": edu_comment
                    })
            except Exception as e:
                continue

        # 密集度が最も低く（綺麗に重なり合っており）、かつエネルギーが詰まっている上位5銘柄を抽出
        if candidates:
            sorted_candidates = sorted(candidates, key=lambda x: x["congestion_width"])
            priority_candidates = sorted_candidates[:5]

        # 毎朝のEarly（学習・教材特化型）メール送信
        notify_early_watch(priority_candidates, latest_date)

    except Exception as e:
        print(f"【エラーログ】Early Watch 監視中に例外が発生しました: {e}")


if __name__ == "__main__":
    main()
