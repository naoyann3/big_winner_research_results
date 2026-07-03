# early_regime_screener.py (Version 1.0 - Early Watch - Fixed)
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
        d["ma25_slope_prev"] = d["ma25_slope"].shift(1)
        
        # 3本MA of 収縮幅（密集度）
        d["ma_max"] = d[["ma25", "ma75", "ma200"]].max(axis=1)
        d["ma_min"] = d[["ma25", "ma75", "ma200"]].min(axis=1)
        d["ma_mean"] = d[["ma25", "ma75", "ma200"]].mean(axis=1)
        d["ma_congestion_width_pct"] = (d["ma_max"] - d["ma_min"]) / d["ma_mean"] * 100
        
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


def notify_early_watch(candidates: list[dict], date_str: str):
    """
    毎朝、移動平均大収縮・トレンド転換初期の「お宝予備軍」を美しく通知します
    """
    if not candidates:
        print("本日のEarly（移動平均大収縮）予備軍は0件です。通知をスキップします。")
        return

    if not (GMAIL_USER and GMAIL_PASS and NOTIFICATION_EMAIL):
        print("警告: メールの認証情報、または通知先アドレスが未設定です。")
        return

    # インポート
    from state5_explainable_engine import State5ExplainableEngine

    msg = MIMEMultipart()
    msg["From"] = f"{SENDER_NAME} <{GMAIL_USER}>"
    msg["To"] = NOTIFICATION_EMAIL
    msg["Subject"] = f"【Early Watch】{date_str} 移動平均大収縮・予備軍 {len(candidates)} 銘柄"

    body = f"# 💡 【Early Watch】{date_str} トレンド転換初期・予備軍リスト\n"
    body += "移動平均線が極限に密集し、今まさに25日線が上向き（トレンド発生）に反転したばかりの、最も新鮮なもみ合い株です。\n"
    body += "買ってすぐに暴騰するものではありませんが、チャートの『呼吸（ブレイクに向けたエネルギー充填）』を観察・学習するのに最も適した精鋭たちです。\n"
    body += "----------------------------------------\n\n"

    for idx, c in enumerate(candidates, 1):
        links = State5ExplainableEngine.get_chart_links(c["ticker"])["all_markdown"]
        body += f"## {idx}. {c['name']} ({c['ticker']}) {links}\n"
        body += f"### 【状態】: **★★★★★ 移動平均大収縮 ＆ トレンド反転初日**\n"
        body += f"*   **MA密集度**: **{c['congestion_width']:.2f}%** (基準: 5%以下)\n"
        body += f"*   **RSI(14)** : {c['rsi14']:.1f}% / BB幅: {c['bb_width']:.1f}% / 出来高比率: {c['vol_ratio']:.2f}倍\n"
        body += f"*   **52週高値からの下落率**: {abs(c['dist_52w']):.1f}%\n"
        body += f"*   **【明日以降の学習・定点観測ToDo】**:\n"
        body += f"     □ このもみ合い（MA収縮）が何日間、この価格帯で継続するか（エネルギー充填の観察）\n"
        body += f"     □ 20日平均の『2倍〜3倍を超える突発的な出来高急増（狼煙：State 3〜4）』が発生するか監視\n"
        body += f"     □ 75日移動平均線（支持帯: {c['ma75']:.1f}円）を完全に割り込んでトレンド崩壊しないか確認\n"
        body += "----------------------------------------\n\n"

    body += "\n※本メールは、チャートの『呼吸』や移動平均線の収縮力学を学び、相場感を養うための研究用レポートです。投資の勉強材料としてTradingViewのリンクから実際の形を確認し、イメージを膨らませてください。\n"

    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASS)
        server.send_message(msg)
    print("毎朝のEarly（予備軍）メールを正常に送信しました。")


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
        priority_candidates = []  # 【安全設計】：ここで最初から空リストで初期化しておきます
        latest_date = None

        print(f"=== Early Watch 予備軍スキャンの稼働を開始します (対象: {len(tickers)} 銘柄) ===")

        for idx, t in enumerate(tickers):
            price_path = PRICES_DIR / f"{t}.csv"
            if not price_path.exists():
                continue

            try:
                df_raw = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()
                if len(df_raw) < 150:
                    continue

                d = EarlyStateEngine.calculate_early_indicators(df_raw)
                row = d.iloc[-1]
                
                if latest_date is None:
                    latest_date = d.index[-1].strftime("%Y-%m-%d")

                # 最低流動性（売買代金）チェック (1,000万円以上)
                if row["turnover_avg20_million"] < TH_MIN_TURNOVER:
                    continue

                # 予備軍の定義（すでに25日線が上を向いている収縮銘柄もすべてスコアリング対象にします）
                if (
                    row["ma_congestion_width_pct"] <= 5.0
                    and row["ma25_slope"] > 0
                    and row["bb_width"] <= 10.0
                    and row["vol_ratio_20"] <= 1.5
                ):
                    candidates.append({
                        "ticker": t,
                        "name": name_map.get(t, t),
                        "congestion_width": row["ma_congestion_width_pct"],
                        "rsi14": row["rsi14"],
                        "bb_width": row["bb_width"],
                        "vol_ratio": row["vol_ratio_20"],
                        "dist_to_52w_high": row["dist_to_52w_high"],
                        "dist_52w": row["dist_to_52w_high"],
                        "ma75": row["ma75"]
                    })
            except Exception:
                continue

        # 【修正・インデント変更】：forループが完全に「終わった後」に、1回だけソートと上位5社抽出を実行します
        if candidates:
            sorted_candidates = sorted(candidates, key=lambda x: x["congestion_width"])
            priority_candidates = sorted_candidates[:5]

        # 毎朝のEarlyメール送信
        notify_early_watch(priority_candidates, latest_date)

    except Exception as e:
        print(f"【エラーログ】Early Watch 監視中に例外が発生しました: {e}")


if __name__ == "__main__":
    main()
