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
SENDER_NAME = config.get("email", {}).get("sender_name", "Sniper OS")

# 閾値
TH_MIN_TURNOVER = config.get("thresholds", {}).get("min_daily_turnover_million", 10.0)
TH_VOL_LIMIT = config.get("thresholds", {}).get("vol_ratio_limit", 0.70)
TH_BB_LIMIT = config.get("thresholds", {}).get("bb_width_limit", 10.0)
TH_RSI_MIN = config.get("thresholds", {}).get("rsi_min", 40.0)
TH_RSI_MAX = config.get("thresholds", {}).get("rsi_max", 60.0)
TH_MA75_DEV = config.get("thresholds", {}).get("ma75_dev_limit", 3.0)

# 配点
WEIGHT_STATE5 = config.get("scoring_weights", {}).get("state5", 20)
WEIGHT_MA75 = config.get("scoring_weights", {}).get("ma75_dev", 20)
WEIGHT_VOL_SHRINK = config.get("scoring_weights", {}).get("vol_shrink", 20)
WEIGHT_BB_SHRINK = config.get("scoring_weights", {}).get("bb_shrink", 15)
WEIGHT_RSI = config.get("scoring_weights", {}).get("rsi", 10)
WEIGHT_DIST_52W = config.get("scoring_weights", {}).get("dist_to_52w_high", 10)
WEIGHT_PO = config.get("scoring_weights", {}).get("perfect_order", 5)

PRIORITY_COUNT = config.get("notification", {}).get("priority_count", 5)
DISPLAY_NAME = config.get("notification", {}).get("display_name", "Gold Watch")

UNIVERSE_CSV = Path("universe.csv")
PRICES_DIR = Path("data_cache/prices")
FUND_DIR = Path("data_cache/fundamentals")


def normalize_ticker(raw: str) -> str:
    ticker = str(raw).strip().upper()
    if not ticker:
        return ticker
    if "." not in ticker and not ticker.isdigit():
        ticker = f"{ticker}.T"
    return ticker


class MarketStateEngine:
    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["ma25"] = d["Close"].rolling(25).mean()
        d["ma75"] = d["Close"].rolling(75).mean()
        d["ma200"] = d["Close"].rolling(200).mean()
        
        d["ma25_slope"] = d["ma25"].pct_change(5) * 100
        d["ma75_slope"] = d["ma75"].pct_change(5) * 100
        d["ma75_dev"] = (d["Close"] - d["ma75"]) / d["ma75"] * 100
        
        d["vol_avg20"] = d["Volume"].rolling(20).mean()
        d["vol_ratio_20"] = d["Volume"] / d["vol_avg20"]
        d["turnover_avg20_million"] = ((d["Close"] * d["Volume"]) / 1_000_000).rolling(20).mean()
        
        std20 = d["Close"].rolling(20).std()
        d["bb_width"] = (std20 * 4) / d["ma25"] * 100
        d["bb_width_min60"] = d["bb_width"].rolling(60).min()
        
        # ATR比率
        high_low = d["High"] - d["Low"]
        high_cp = (d["High"] - d["Close"].shift(1)).abs()
        low_cp = (d["Low"] - d["Close"].shift(1)).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        d["atr_ratio"] = (tr.rolling(14).mean() / d["Close"]) * 100
        
        delta = d["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        d["rsi14"] = 100 - (100 / (1 + (gain / np.where(loss > 0, loss, 1.0))))
        
        d["high_52w"] = d["High"].rolling(250, min_periods=50).max()
        d["low_52w"] = d["Low"].rolling(250, min_periods=50).min()
        d["dist_to_52w_high"] = (d["Close"] - d["high_52w"]) / d["high_52w"] * 100
        d["dist_to_52w_low"] = (d["Close"] - d["low_52w"]) / d["low_52w"] * 100
        d["high_20d"] = d["High"].shift(1).rolling(20).max()

        return d

    @staticmethod
    def simulate_state_machine(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        states = []
        state_durations = []
        
        current_state = 0
        state_days = 0
        last_high = 0.0
        
        for idx in range(len(d)):
            row = d.iloc[idx]
            close = row["Close"]
            bb_width = row["bb_width"]
            bb_min = row["bb_width_min60"]
            ma25_slope = row["ma25_slope"]
            rsi14 = row["rsi14"]
            vol_ratio = row["vol_ratio_20"]
            high_20d = row["high_20d"]
            
            if pd.isna(bb_width) or pd.isna(ma25_slope) or pd.isna(rsi14) or pd.isna(vol_ratio):
                states.append(0)
                state_durations.append(0)
                continue
                
            last_high = max(last_high, row["High"]) if current_state > 0 else row["High"]
            
            if current_state > 0 and close < last_high * 0.90:
                current_state = 0
                state_days = 0
                last_high = row["High"]

            next_state = current_state
            
            if current_state == 0:
                if bb_width <= bb_min * 1.05: next_state = 1
            elif current_state == 1:
                if ma25_slope > 0 and rsi14 >= 50.0: next_state = 2
            elif current_state == 2:
                if vol_ratio >= 2.0: next_state = 3
            elif current_state == 3:
                if vol_ratio >= 3.0 and close > row["Open"]: next_state = 4
            elif current_state == 4:
                if close < row["Open"] and vol_ratio < 1.0: next_state = 5
            elif current_state == 5:
                if close > high_20d and vol_ratio >= 1.5: next_state = 6
                
            if next_state != current_state:
                current_state = next_state
                state_days = 1
            else:
                state_days += 1
                
            states.append(current_state)
            state_durations.append(state_days)
            
        d["current_state"] = states
        d["state_days"] = state_durations
        return d


def score_and_comment_candidate(latest_row: pd.Series) -> tuple[int, list[str]]:
    score = 0
    comments = []
    
    if int(latest_row["current_state"]) == 5:
        score += WEIGHT_STATE5
    
    ma75_dev = latest_row["ma75_dev"]
    if abs(ma75_dev) <= TH_MA75_DEV:
        score += WEIGHT_MA75
        comments.append("MA75支持確認")
    
    vol_ratio = latest_row["vol_ratio_20"]
    if vol_ratio <= TH_VOL_LIMIT:
        score += WEIGHT_VOL_SHRINK
        comments.append("出来高収縮継続")
    
    bb_width = latest_row["bb_width"]
    if bb_width <= TH_BB_LIMIT:
        score += WEIGHT_BB_SHRINK
        comments.append("ボラティリティ低下")
    
    rsi14 = latest_row["rsi14"]
    if TH_RSI_MIN <= rsi14 <= TH_RSI_MAX:
        score += WEIGHT_RSI
        comments.append("RSI適正")
    
    dist_52w = latest_row["dist_to_52w_high"]
    if abs(dist_52w) <= 20.0:
        score += WEIGHT_DIST_52W
        comments.append(f"52週高値まで {abs(dist_52w):.1f}%")
    
    ma25 = latest_row["ma25"]
    ma75 = latest_row["ma75"]
    ma200 = latest_row["ma200"]
    if ma25 > ma75 > ma200:
        score += WEIGHT_PO
        comments.append("上昇パーフェクトオーダー維持")
        
    if latest_row["Close"] > ma200:
        comments.append("長期移動平均線上")
        
    if latest_row["Volume"] < latest_row["vol_avg20"] * 0.5:
        comments.append("需給改善中")

    return score, comments


def notify_state5_watch(candidates: list[dict], date_str: str):
    if not candidates:
        print("本日のState 5優先候補は0件です。通知をスキップします。")
        return

    if not (GMAIL_USER and GMAIL_PASS and NOTIFICATION_EMAIL):
        print("警告: メールの認証情報、または通知先アドレスが未設定です。")
        return

    msg = MIMEMultipart()
    msg["From"] = f"{SENDER_NAME} <{GMAIL_USER}>"
    msg["To"] = NOTIFICATION_EMAIL
    msg["Subject"] = f"【State5 Watch】{date_str} 優先候補 {len(candidates)} 銘柄"

    body = f"# 【{DISPLAY_NAME}】{date_str} 優先度順リスト\n"
    body += "過去の大化け株データに基づく評価値（100点満点）の上位優先候補です。\n"
    body += "----------------------------------------\n\n"

    for idx, c in enumerate(candidates, 1):
        stars = "★" * max(1, int(c["score"] / 20))
        body += f"### {idx}. {c['name']} ({c['ticker']})\n"
        body += f"**スコア: {c['score']}点** ({stars})\n"
        body += f"*   **現在状態**: State {c['state']} (滞在: {c['days_in_state']}日目)\n"
        body += f"*   **MA75乖離**: {c['ma75_dev']:+.1f}%\n"
        body += f"*   **RSI(14)**: {c['rsi14']:.1f}\n"
        body += f"*   **BB幅**: {c['bb_width']:.1f}%\n"
        body += f"*   **出来高比率**: {c['vol_ratio']:.2f}\n"
        body += f"*   **定型コメント**: {', '.join(c['comments'])}\n"
        body += "----------------------------------------\n\n"

    body += "\n※本システムは客観的データに基づき期待値の高い候補を提示しています。最終的な投資判断は必ずチャートを確認した上で行ってください。\n"

    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASS)
        server.send_message(msg)
    print("毎朝のState 5優先候補メールを正常に送信しました。")


def main():
    try:
        if not UNIVERSE_CSV.exists():
            print(f"宇宙ファイル {UNIVERSE_CSV} が存在しません。処理を中断します。")
            return

        df_uni = pd.read_csv(UNIVERSE_CSV)
        tickers = df_uni["ticker"].dropna().tolist()
        tickers = [normalize_ticker(t) for t in tickers]
        
        name_map = dict(zip(df_uni["ticker"].map(normalize_ticker), df_uni["name"]))

        candidates = []
        latest_date = None

        print(f"=== State 5 監視＆スコアリングシステムの稼働を開始します (対象: {len(tickers)} 銘柄) ===")

        for idx, t in enumerate(tickers):
            price_path = PRICES_DIR / f"{t}.csv"
            if not price_path.exists():
                continue

            try:
                df_raw = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()
                if len(df_raw) < 150:
                    continue

                df_ind = MarketStateEngine.calculate_indicators(df_raw)
                df_sim = MarketStateEngine.simulate_state_machine(df_ind)
                
                latest_row = df_sim.iloc[-1]
                latest_state = int(latest_row["current_state"])
                
                if latest_date is None:
                    latest_date = df_sim.index[-1].strftime("%Y-%m-%d")

                if latest_row["turnover_avg20_million"] < TH_MIN_TURNOVER:
                    continue

                # --- 【テスト用】：条件を if True: にして強制的に全銘柄をスコアリング ---
                if True:
                    score, comments = score_and_comment_candidate(latest_row)
                    
                    candidates.append({
                        "ticker": t,
                        "name": name_map.get(t, t),
                        "score": score,
                        "state": latest_state,
                        "days_in_state": int(latest_row["state_days"]),
                        "close": float(latest_row["Close"]),
                        "ma75_dev": latest_row["ma75_dev"],
                        "rsi14": latest_row["rsi14"],
                        "bb_width": latest_row["bb_width"],
                        "vol_ratio": latest_row["vol_ratio_20"],
                        "comments": comments,
                        # 教師データ用の追加テクニカル特徴量
                        "dist_to_52w_high": latest_row["dist_to_52w_high"],
                        "dist_to_52w_low": latest_row["dist_to_52w_low"],
                        "ma25_slope": latest_row["ma25_slope"],
                        "atr_ratio": latest_row["atr_ratio"]
                    })
            except Exception:
                continue

        # スコアの高い順にソートして、上位5銘柄を抽出
        sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        priority_candidates = sorted_candidates[:PRIORITY_COUNT]

        # 毎朝のメール送信
        notify_state5_watch(priority_candidates, latest_date)
        
        # ==========================================
        # ★【Version 7 新設】：自律学習・成績管理システムの自動フック ★
        # ==========================================
        try:
            print("\n=== Version 7: 研究データ収集・成績管理システムを自動起動します ===")
            
            # 1. 市場環境（地合い）の自動判定
            from market_environment import MarketEnvironmentManager
            market_env = MarketEnvironmentManager.get_current_environment(latest_date)
            print(f"  [市場環境] TOPIX終値: {market_env['topix_close']:.1f} (地合い: {market_env['market_state_topix']})")
            
            # 2. 教師データ（履歴）のロギング
            from state5_history_logger import State5HistoryLogger
            State5HistoryLogger.log_candidates(candidates, latest_date, market_env, config)
            
            # 3. 過去シグナルの成績自動追跡（採点）
            from performance_tracker import PerformanceTracker
            PerformanceTracker.track_and_score_history(config)
            
            # 4. 実績評価レポート（Champion Report）の自動生成
            from champion_report import ChampionReportGenerator
            ChampionReportGenerator.generate_report(config)
            
            print("=== Version 7: すべての研究データ更新・成績管理処理が正常に完了しました ===")
            
        except Exception as e:
            print(f"【エラーログ】Version 7 モジュール実行中に例外が発生しました: {e}")

    except Exception as e:
        print(f"【エラーログ】監視システム稼働中に致命的な例外が発生しました: {e}")


if __name__ == "__main__":
    main()
