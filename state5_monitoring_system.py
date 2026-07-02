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
        comments.append("出来高収縮完成")
    
    bb_width = latest_row["bb_width"]
    if bb_width <= TH_BB_LIMIT:
        score += WEIGHT_BB_SHRINK
        comments.append("BB収縮完成")
    
    rsi14 = latest_row["rsi14"]
    if TH_RSI_MIN <= rsi14 <= TH_RSI_MAX:
        score += WEIGHT_RSI
        comments.append("RSI適正")
    
    dist_52w = latest_row["dist_to_52w_high"]
    if abs(dist_52w) <= 20.0:
        score += WEIGHT_DIST_52W
        comments.append(f"52週高値近接")
    
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
    【Version 7.9 意思決定支援・3階層レイアウト特化版】：
    Layer 1（3秒）、Layer 2（15秒）、Layer 3（詳細）の3階層に完全に情報設計を分離。
    メール最上部に「今日のAI総括」、最下部に「今日のAction Log」を配置。
    """
    if not candidates:
        print("本日のState 5優先候補は0件です。通知をスキップします。")
        return

    if not (GMAIL_USER and GMAIL_PASS and NOTIFICATION_EMAIL):
        print("警告: メールの認証情報、または通知先アドレスが未設定です。")
        return

    from state5_explainable_engine import State5ExplainableEngine
    
    # 地合いの星評価と期待値
    star_title, env_desc, stats_str = State5ExplainableEngine.get_market_env_expectancy_v71(market_state, config)

    # ④：今日の一言総括（AI総括）の自動生成
    ai_summary_str = State5ExplainableEngine.generate_ai_summary(candidates, market_state)

    # ⑥：今日のAction Log（最下部のToDoチェックリスト）の自動生成
    action_log_str = State5ExplainableEngine.generate_action_log(candidates)

    msg = MIMEMultipart()
    msg["From"] = f"{SENDER_NAME} <{GMAIL_USER}>"
    msg["To"] = NOTIFICATION_EMAIL
    msg["Subject"] = f"【State5 Watch】{date_str} 優先候補 {len(candidates)} 銘柄"

    # ヘッダー構築
    body = f"# 【{DISPLAY_NAME}】{date_str} 意思決定支援レポート\n"
    body += "※情報量よりも「人間が1分以内で監視対象を決定できること」を最優先に設計されたレポートです。\n"
    body += "----------------------------------------\n"
    body += f"### 📝 【本日のAI総括 (200文字要約)】\n"
    body += f"{ai_summary_str}\n"
    body += "----------------------------------------\n"
    
    # ⑤ 最終コメント：「本日の最重要監視銘柄TOP3」の自動生成（1分要約）
    top3_str = ""
    for idx, c in enumerate(candidates[:3], 1):
        top3_str += f"  {idx}位: **{c['name']} ({c['ticker']})** ➔ 総合 {c['evaluation_score']:.1f}点 ({c['rank']}) / {c['action_star']}\n"
        top3_str += f"        (一致率: {c['type0_match_rate']}%。{c['maturity_short_desc']}。{c['comments'][0]}等)\n"

    body += "### 💡 【本日の最重要監視銘柄 TOP3 （1分要約）】\n"
    body += top3_str
    body += "----------------------------------------\n\n"
    
    body += f"### ■ 本日の相場環境判定: 【 {star_title} 】\n"
    body += f"*   **地合い状況**: {env_desc}\n"
    body += f"**【現在の地合いにおける、過去5,487件の実績期待値】**:\n{stats_str}\n"
    body += "----------------------------------------\n\n"

    # 各銘柄詳細（Layer 1 〜 Layer 3 の3階層構造へ大リファクタリング）
    for idx, c in enumerate(candidates, 1):
        body += f"## {idx}. {c['name']} ({c['ticker']}) {c['links']}\n"
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 【Layer 1 (3秒判定エリア)】
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        body += "### ━━━━━━━━━━━━━━━━━━\n"
        body += "### 🔴 Layer 1：本日の優先度とアクション指示\n"
        body += "### ━━━━━━━━━━━━━━━━━━\n"
        body += f"  ・総合評価スコア: **{c['evaluation_score']:.1f}点** (ベース点: {c['score']} / ランク: {c['rank']})\n"
        body += f"  ・現在の監視優先: **{c['action_star']}**\n"
        
        # ②：「昨日から何が変わったか（前日差分）」を最優先表示
        body += f"{c['diff_text']}"
        
        # ⑥：「今日やること」のToDoリスト
        body += "  ・【今日のToDo行動指針】\n"
        for t_item in c["todo"]:
            body += f"     {t_item}\n"
        body += "### ━━━━━━━━━━━━━━━━━━\n\n"
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 【Layer 2 (15秒判定エリア)】
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        body += "### 🟡 Layer 2：なぜそう判断したか（強みと注意点）\n"
        body += f"  ・推定チャート形状: **{c['chart_pattern']}**\n"
        body += f"  ・状態遷移成熟度  : {c['maturity_desc']}\n"
        body += f"  ・信頼度(Confidence): **{c['confidence']}%** ({c['confidence_stars']}) / Type0一致率: **{c['type0_match_rate']}%** ({c['match_stars']})\n"
        body += f"  ・過去類似DNA実績 : {c['similar_stats_str']}\n"
        
        # Avoid時の理由説明
        if "見送り" in c["action_star"]:
            body += f"  ・{c['avoid_desc']}\n"
            
        body += "\n"
        body += "  ・【買う理由（強み）】\n"
        for p in c["pros"]:
            body += f"     - {p}\n"
        body += "\n"
        body += "  ・【注意点（弱み）】\n"
        for con in c["cons"]:
            body += f"     * {con}\n"
        body += "\n"
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 【Layer 3 (詳細データエリア - 必要時のみ確認)】
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        body += "### 🔵 Layer 3：詳細データ・AIコメント\n"
        body += f"  [基本データ]: 終値: {c['close']:.1f} 円 (MA75乖離: {c['ma75_dev']:+.1f}%) / RSI(14): {c['rsi14']:.1f}% / BB幅: {c['bb_width']:.1f}%\n"
        body += "  [獲得加点内訳 (獲得点数 / 配点)]:\n"
        for item, (gain, max_p) in c["score_details"].items():
            body += f"     - {item:12s}: {gain:2d} / {max_p:2d}\n"
        body += "\n"
        body += f"  {c['ai_comment']}\n"
        body += "----------------------------------------\n\n"

    # メールの最下部に「今日のAction Log」を配置
    body += f"\n{action_log_str}\n\n"
    body += "\n※本システムは未来の株価を断定・予言するものではありません。期待値の高い局面にいる銘柄を自動選別することで、人間の分析・判断時間を極限まで削減することを目的に設計されています。最終判断は必ずチャートを確認の上、ご自身の規律に従って行ってください。\n"

    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASS)
        server.send_message(msg)
    print("毎朝のState 5優先候補（意思決定支援プロファイル型）メールを正常に送信しました。")


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

                # --- 【本番運用仕様に戻しました】：State 5 のみスキャン ---
                if latest_state == 5:
                    score, comments = score_and_comment_candidate(latest_row)
                    
                    # --- 説明可能パラメータの自動算出 ---
                    details, deductions = State5ExplainableEngine.get_score_details_and_deductions(latest_row, config)
                    type0_match = State5ExplainableEngine.get_type0_matching_rate(latest_row)
                    maturity_desc = State5ExplainableEngine.get_state5_maturity(int(latest_row["state_days"]))
                    confidence, conf_rank, overall_rank = State5ExplainableEngine.get_confidence_and_rank(score, type0_match, market_state)
                    
                    # 【Version 7.8新設】：チャート形状・強み・注意点の自動分析
                    chart_pattern = State5ExplainableEngine.get_chart_pattern(df_raw)
                    pros, cons = State5ExplainableEngine.get_pros_and_cons(latest_row)
                    
                    # 【Version 7.8新設】：優先度星評価
                    action_star, action_short = State5ExplainableEngine.get_action_recommendation_v71(score, confidence, int(latest_row["state_days"]))
                    
                    # 【Version 7.8新設】：お宝TradingView/株探/SBI証券ダイレクトリンク自動生成
                    links_dict = State5ExplainableEngine.get_chart_links(t)
                    
                    # 【Version 7.8新設】：過去類似統計 ＆ Avoid統計理由の自動算出
                    similar_stats_str, sim_stats = State5ExplainableEngine.get_similar_history_stats(type0_match, market_state, config)
                    avoid_desc = State5ExplainableEngine.explain_avoid_reason(int(latest_row["state_days"]))
                    
                    # 【Version 7.8新設】：「今日やること」のToDoリスト自動生成
                    todo = State5ExplainableEngine.generate_daily_todo(latest_row, action_short, chart_pattern)
                    
                    # 【Version 7.9新設】：「昨日から何が変わったか（前日差分）」の自動計算
                    current_data_for_diff = {
                        "score": score,
                        "vol_ratio": float(latest_row["vol_ratio_20"]),
                        "days_in_state": int(latest_row["state_days"])
                    }
                    diff_text = State5ExplainableEngine.get_previous_diff(t, latest_date, current_data_for_diff, config)
                    
                    # 自然言語AIコメントにチャート形状判定と簡潔さを適用
                    ai_comment = State5ExplainableEngine.get_natural_ai_comment(latest_row, type0_match, chart_pattern)
                    
                    # 1分要約用の簡易成熟度
                    maturity_short_desc = f"State 5に入って {int(latest_row['state_days'])}日目"
                    
                    # 星評価
                    confidence_stars = State5ExplainableEngine.get_star_rating(confidence)
                    match_stars = State5ExplainableEngine.get_star_rating(type0_match)
                    
                    # 最終優先順位スコアの算出
                    evaluation_score = State5ExplainableEngine.calculate_evaluation_score(score, type0_match, confidence, int(latest_row["state_days"]))
                    
                    candidates.append({
                        "ticker": t,
                        "name": name_map.get(t, t),
                        "score": score,
                        "evaluation_score": evaluation_score,  # 重み付けされた最終優先順位スコア
                        "rank": overall_rank,
                        "state": latest_state,
                        "days_in_state": int(latest_row["state_days"]),
                        "close": float(latest_row["Close"]),
                        "ma75_dev": latest_row["ma75_dev"],
                        "rsi14": latest_row["rsi14"],
                        "bb_width": latest_row["bb_width"],
                        "vol_ratio": latest_row["vol_ratio_20"],
                        "comments": comments,
                        
                        # 意思決定支援パラメータ
                        "chart_pattern": chart_pattern,
                        "pros": pros,
                        "cons": cons,
                        "action_star": action_star,
                        "maturity_short_desc": maturity_short_desc,
                        "similar_stats_str": similar_stats_str,
                        "similar_win": sim_stats["win_rate"],
                        "avoid_desc": avoid_desc,
                        "todo": todo,
                        "links": links_dict["all_markdown"],
                        "confidence_stars": confidence_stars,
                        "match_stars": match_stars,
                        "diff_text": diff_text,
                        
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
            except Exception as e:
                continue

        # 重み付けされた「最終優先順位スコア」でソート
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
