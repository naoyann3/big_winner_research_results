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


def notify_state5_watch(candidates: list[dict], date_str: str, market_state: str):
    """
    【Version 7.6 意思決定支援特化版】：
    1分以内の状況把握を最優先し、行動推奨、強み・注意点の簡潔な箇条書き、
    および「本日の最重要監視TOP3」を最上部に配置したプロファイルレポートを送信します。
    """
    if not candidates:
        print("本日のState 5優先候補は0件です。通知をスキップします。")
        return

    if not (GMAIL_USER and GMAIL_PASS and NOTIFICATION_EMAIL):
        print("警告: メールの認証情報、または通知先アドレスが未設定です。")
        return

    from state5_explainable_engine import State5ExplainableEngine
    
    env_desc, stats_str = State5ExplainableEngine.get_market_expectancy_and_stats(market_state, config)

    msg = MIMEMultipart()
    msg["From"] = f"{SENDER_NAME} <{GMAIL_USER}>"
    msg["To"] = NOTIFICATION_EMAIL
    msg["Subject"] = f"【State5 Watch】{date_str} 優先候補 {len(candidates)} 銘柄"

    # ⑤ 最終コメント：「本日の最重要監視銘柄TOP3」の自動生成（1分要約）
    top3_str = ""
    for idx, c in enumerate(candidates[:3], 1):
        stars = "★" * max(1, int(c["score"] / 20))
        top3_str += f"  {idx}位: **{c['name']} ({c['ticker']})** ➔ 総合 {c['score']}点 ({stars}) / {c['action']}\n"
        top3_str += f"        (一致率 {c['type0_match_rate']}%。{c['maturity_short_desc']}。{c['comments'][0]}等)\n"

    # ヘッダー構築
    body = f"# 【{DISPLAY_NAME}】{date_str} 意思決定支援レポート\n"
    body += "※情報量よりも「人間が1分以内で監視対象を決定できること」を最優先に設計されたレポートです。\n"
    body += "----------------------------------------\n"
    body += "### 💡 【本日の最重要監視銘柄 TOP3 （1分要約）】\n"
    body += top3_str
    body += "----------------------------------------\n\n"
    
    body += f"### ■ 本日の相場環境判定: 【 {market_state} 】\n"
    body += f"*   **地合い状況**: {env_desc}\n"
    body += f"**【現在の地合いにおける、過去5,487件の実績期待値】**:\n{stats_str}\n"
    body += "----------------------------------------\n\n"

    # 各銘柄詳細
    for idx, c in enumerate(candidates, 1):
        stars = "★" * max(1, int(c["score"] / 20))
        body += f"## {idx}. {c['name']} ({c['ticker']})\n"
        body += f"### 【評価】: {c['score']}点 (ランク: {c['rank']}) ➔ 【 {c['action']} 】\n"
        body += f"**信頼度 (Confidence): {c['confidence']}% (ランク: {c['conf_rank']}) / Type 0一致率: {c['type0_match_rate']}%**\n\n"
        
        # チャート形状
        body += f"*   **推定チャート形状**: **{c['chart_pattern']}**\n"
        body += f"*   **状態遷移成熟度**: {c['maturity_desc']}\n\n"
        
        # ② 強み（買う理由）＆ 注意点（弱み）を簡潔に
        body += "**【買う理由（強み）】**\n"
        for p in c["pros"]:
            body += f"  - {p}\n"
        body += "\n"
        
        body += "**【注意点（弱み）】**\n"
        for con in c["cons"]:
            body += f"  * {con}\n"
        body += "\n"
        
        # 基本データ
        body += "【基本テクニカル】\n"
        body += f"  終値: {c['close']:.1f} 円 (MA75乖離: {c['ma75_dev']:+.1f}%) / RSI(14): {c['rsi14']:.1f}% / BB幅: {c['bb_width']:.1f}% / 出来高比率: {c['vol_ratio']:.2f}倍\n\n"
        
        # スコア内訳
        body += "【加点内訳 (獲得点数 / 配点)】\n"
        for item, (gain, max_p) in c["score_details"].items():
            body += f"  - {item:12s}: {gain:2d} / {max_p:2d}\n"
        body += "\n"
        
        # 自然言語AIコメント
        body += f"{c['ai_comment']}\n"
        body += "----------------------------------------\n\n"

    body += "\n※本システムは未来の株価を断定・予言するものではありません。期待値の高い局面にいる銘柄を自動選別することで、人間の分析・判断時間を極限まで削減することを目的に設計されています。最終判断は必ずチャートを確認の上、ご自身の規律に従って行ってください。\n"

    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASS)
        server.send_message(msg)
    print("毎朝のState 5優先候補（説明可能プロファイル型）メールを正常に送信しました。")


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

        from state5_explainable_engine import State5ExplainableEngine
        from market_environment import MarketEnvironmentManager

        first_ticker = tickers[0]
        try:
            df_dummy = pd.read_csv(PRICES_DIR / f"{first_ticker}.csv", index_col=0, parse_dates=True)
            latest_date = df_dummy.index[-1].strftime("%Y-%m-%d")
        except Exception:
            latest_date = datetime.now().strftime("%Y-%m-%d")

        market_env = MarketEnvironmentManager.get_current_environment(latest_date)
        market_state = market_env["market_state_topix"]

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

                if latest_row["turnover_avg20_million"] < TH_MIN_TURNOVER:
                    continue

                # 必須判定: State 5 であること
                if latest_state == 5:
                    score, comments = score_and_comment_candidate(latest_row)
                    
                    # --- 説明可能パラメータの自動算出 ---
                    details, deductions = State5ExplainableEngine.get_score_details_and_deductions(latest_row, config)
                    type0_match = State5ExplainableEngine.get_type0_matching_rate(latest_row)
                    maturity_desc = State5ExplainableEngine.get_state5_maturity(int(latest_row["state_days"]))
                    confidence, conf_rank, overall_rank = State5ExplainableEngine.get_confidence_and_rank(score, type0_match, market_state)
                    
                    # 【Version 7.6新設】：チャート形状自動分析
                    chart_pattern = State5ExplainableEngine.get_chart_pattern(df_raw)
                    
                    # 【Version 7.6新設】：買う理由（強み）＆ 注意点（弱み）を自動抽出
                    pros, cons = State5ExplainableEngine.get_pros_and_cons(latest_row)
                    
                    # 【Version 7.6新設】：行動推奨（4段階）
                    action = State5ExplainableEngine.get_action_recommendation(score, confidence, int(latest_row["state_days"]))
                    
                    # 自然言語AIコメントにチャート形状判定を統合
                    ai_comment = State5ExplainableEngine.get_natural_ai_comment(latest_row, type0_match, chart_pattern)
                    
                    # 1分要約用の簡易成熟度
                    maturity_short_desc = f"State 5に入って {int(latest_row['state_days'])}日目"
                    
                    # ③：Type 0 一致率を加重した「総合評価値（evaluation_score）」の算出
                    # これにより、同じ100点や95点でも、一致率が高いものが確実に最上位にソートされます
                    evaluation_score = score + (type0_match * 0.1)
                    
                    candidates.append({
                        "ticker": t,
                        "name": name_map.get(t, t),
                        "score": score,
                        "evaluation_score": evaluation_score,  # 重み付け用スコア
                        "rank": overall_rank,
                        "state": latest_state,
                        "days_in_state": int(latest_row["state_days"]),
                        "close": float(latest_row["Close"]),
                        "ma75_dev": latest_row["ma75_dev"],
                        "rsi14": latest_row["rsi14"],
                        "bb_width": latest_row["bb_width"],
                        "vol_ratio": latest_row["vol_ratio_20"],
                        "comments": comments,
                        
                        # 7.6 新設の意思決定支援パラメータ
                        "chart_pattern": chart_pattern,
                        "pros": pros,
                        "cons": cons,
                        "action": action,
                        "maturity_short_desc": maturity_short_desc,
                        
                        # 説明可能パラメータ
                        "score_details": details,
                        "deductions": deductions,
                        "type0_match_rate": type0_match,
                        "maturity_desc": maturity_desc,
                        "confidence": confidence,
                        "conf_rank": conf_rank,
                        "ai_comment": ai_comment,
                        
                        # 教師データ用の追加テクニカル特徴量
                        "dist_to_52w_high": latest_row["dist_to_52w_high"],
                        "dist_to_52w_low": latest_row["dist_to_52w_low"],
                        "ma25_slope": latest_row["ma25_slope"],
                        "atr_ratio": latest_row["atr_ratio"]
                    })
            except Exception:
                continue

        # ③：Type 0 一致率を加重した「総合評価値（evaluation_score）」でソート
        sorted_candidates = sorted(candidates, key=lambda x: x["evaluation_score"], reverse=True)
        priority_candidates = sorted_candidates[:PRIORITY_COUNT]

        # 毎朝の説明可能プロファイルメール送信 (地合いを考慮)
        notify_state5_watch(priority_candidates, latest_date, market_state)
        
        # ==========================================
        # ★【Version 7.0】：自律学習・成績管理システムの自動フック ★
        # ==========================================
        try:
            print("\n=== Version 7: 研究データ収集・成績管理システムを自動起動します ===")
            
            # 1. 教師データ（履歴）のロギング
            from state5_history_logger import State5HistoryLogger
            State5HistoryLogger.log_candidates(candidates, latest_date, market_env, config)
            
            # 2. 過去シグナルの成績自動追跡（採点）
            from performance_tracker import PerformanceTracker
            PerformanceTracker.track_and_score_history(config)
            
            # 3. 実績評価レポート（Champion Report）の自動生成
            from champion_report import ChampionReportGenerator
            ChampionReportGenerator.generate_report(config)
            
            print("=== Version 7: すべての研究データ更新・成績管理処理が正常に完了しました ===")
            
        except Exception as e:
            print(f"【エラーログ】Version 7 モジュール実行中に例外が発生しました: {e}")

    except Exception as e:
        print(f"【エラーログ】監視システム稼働中に致命的な例外が発生しました: {e}")


if __name__ == "__main__":
    main()
