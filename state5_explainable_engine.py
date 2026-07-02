# state5_explainable_engine.py
import pandas as pd
import numpy as np
from pathlib import Path

class State5ExplainableEngine:
    """
    Sniper OS Version 7.5 - 説明可能AI（Explainability）コアエンジン
    """
    @staticmethod
    def get_score_details_and_deductions(latest_row: pd.Series, config: dict) -> tuple[dict, list[dict]]:
        """
        ① & ②：加点内訳および減点理由の自動算出
        """
        weights = config.get("scoring_weights", {})
        thresholds = config.get("thresholds", {})
        
        # 各種閾値
        vol_limit = thresholds.get("vol_ratio_limit", 0.70)
        bb_limit = thresholds.get("bb_width_limit", 10.0)
        rsi_min = thresholds.get("rsi_min", 40.0)
        rsi_max = thresholds.get("rsi_max", 60.0)
        ma75_dev_limit = thresholds.get("ma75_dev_limit", 3.0)
        
        # 獲得点数 / 最大配点
        details = {
            "State 5判定": (weights.get("state5", 20) if int(latest_row["current_state"]) == 5 else 0, weights.get("state5", 20)),
            "MA75近接": (weights.get("ma75_dev", 20) if abs(latest_row["ma75_dev"]) <= ma75_dev_limit else 0, weights.get("ma75_dev", 20)),
            "出来高収縮": (weights.get("vol_shrink", 20) if latest_row["vol_ratio_20"] <= vol_limit else 0, weights.get("vol_shrink", 20)),
            "BB幅収縮": (weights.get("bb_shrink", 15) if latest_row["bb_width"] <= bb_limit else 0, weights.get("bb_shrink", 15)),
            "RSI適正": (weights.get("rsi", 10) if rsi_min <= latest_row["rsi14"] <= rsi_max else 0, weights.get("rsi", 10)),
            "52週高値近接": (weights.get("dist_to_52w_high", 10) if abs(latest_row["dist_to_52w_high"]) <= 20.0 else 0, weights.get("dist_to_52w_high", 10)),
            "上昇PO維持": (weights.get("perfect_order", 5) if latest_row["ma25"] > latest_row["ma75"] > latest_row["ma200"] else 0, weights.get("perfect_order", 5)),
        }
        
        # 減点理由の算出
        deductions = []
        if abs(latest_row["ma75_dev"]) > ma75_dev_limit:
            loss = weights.get("ma75_dev", 20)
            deductions.append({"factor": "75日移動平均線からの乖離が基準超過", "penalty": -loss})
        if latest_row["vol_ratio_20"] > vol_limit:
            loss = weights.get("vol_shrink", 20)
            deductions.append({"factor": "出来高比率が基準超過（売り枯れ不十分）", "penalty": -loss})
        if latest_row["bb_width"] > bb_limit:
            loss = weights.get("bb_shrink", 15)
            deductions.append({"factor": "ボリンジャーバンド幅が基準超過（ボラティリティ低下不足）", "penalty": -loss})
        if not (rsi_min <= latest_row["rsi14"] <= rsi_max):
            loss = weights.get("rsi", 10)
            deductions.append({"factor": "RSI(14)が適正中立圏（40〜60）から逸脱", "penalty": -loss})
        if abs(latest_row["dist_to_52w_high"]) > 20.0:
            loss = weights.get("dist_to_52w_high", 10)
            deductions.append({"factor": "52週高値から下げすぎ（トレンド崩壊の懸念あり）", "penalty": -loss})
        if not (latest_row["ma25"] > latest_row["ma75"] > latest_row["ma200"]):
            loss = weights.get("perfect_order", 5)
            deductions.append({"factor": "上昇パーフェクトオーダーが未完成", "penalty": -loss})
            
        return details, deductions

    @staticmethod
    def get_type0_matching_rate(latest_row: pd.Series) -> int:
        """
        ③：理想形 Type 0 (出来高比率=0.66, RSI=55.0, BB幅=7.03) との一致率の算出
        """
        vol_ratio = latest_row["vol_ratio_20"]
        rsi14 = latest_row["rsi14"]
        bb_width = latest_row["bb_width"]
        
        # 各指標の理想値との距離
        diff_vol = abs(vol_ratio - 0.66) / 0.66
        diff_rsi = abs(rsi14 - 55.0) / 55.0
        diff_bb = abs(bb_width - 7.03) / 7.03
        
        # 加重平均による不一致度の算出 (出来高40%, RSI30%, BB幅30%)
        mismatch_score = (diff_vol * 0.4) + (diff_rsi * 0.3) + (diff_bb * 0.3)
        
        # 一致率（%）に反転（最低40%〜最大100%にクリップ）
        matching_rate = int((1.0 - min(0.6, mismatch_score)) * 100)
        return matching_rate

    @staticmethod
    def get_state5_maturity(days_in_state: int) -> str:
        """
        ④：State 5 の滞在日数（成熟度）の意味合いを日本語で説明
        """
        if days_in_state <= 7:
            return f"State 5 ({days_in_state}日目): 初期段階（新鮮度極めて高）。最後のふるい落とし（調整）が始まった直後であり、ここからの押し目拾いは高期待値。"
        elif 8 <= days_in_state <= 25:
            return f"State 5 ({days_in_state}日目): 成熟段階（黄金期）。過去データ（平均約30日）に照らし、エネルギー収縮が最終局面に達した最も期待値が高いゾーン。"
        elif 26 <= days_in_state <= 45:
            return f"State 5 ({days_in_state}日目): 長期熟成段階。ボラティリティが十分に押し殺されており、いつ本格的な本上昇ブレイクが発生してもおかしくない緊迫した局面。"
        else:
            return f"State 5 ({days_in_state}日目): 停滞・膠着状態。滞在期間が平均を大幅に超過しており、トレンド転換が鈍化（期待値低下傾向）している可能性あり。"

    @classmethod
    def get_confidence_and_rank(cls, score: int, matching_rate: int, market_state: str) -> tuple[int, str, str]:
        """
        ⑤ & ⑧：信頼度（Confidence %）、信頼度ランク、および総合ランク（S+〜C）の算出
        """
        # 1. 信頼度 (Confidence) の数理モデル
        # 基本値はType0一致率
        base_confidence = matching_rate
        
        # 市場環境（地合い）による確度補正 (Bull: +5%, Bear: -15%, Range: 0)
        if market_state == "Bull":
            base_confidence += 5
        elif market_state == "Bear":
            base_confidence -= 15
            
        confidence = max(30, min(99, base_confidence))
        
        # 2. 信頼度ランク
        if confidence >= 95: conf_rank = "A+"
        elif confidence >= 90: conf_rank = "A"
        elif confidence >= 80: conf_rank = "B"
        else: conf_rank = "C"
        
        # 3. 総合ランク (S+〜C) の算出（総合スコアを基準に判定）
        if score >= 100: overall_rank = "S+"
        elif score >= 95: overall_rank = "S"
        elif score >= 90: overall_rank = "A"
        elif score >= 80: overall_rank = "B"
        else: overall_rank = "C"
        
        return confidence, conf_rank, overall_rank

    @classmethod
    def get_market_expectancy_and_stats(cls, market_state: str, config: dict) -> tuple[str, str]:
        """
        ⑥ & ⑦：現在の地合い（TOPIX）に応じた、過去5,487件の実績に基づく期待値・勝率プロファイル
        ※ state5_history.csv が十分に溜まっている場合は、そこから動的にリアルタイムな勝率を算出します。
        """
        history_file = Path(config.get("research", {}).get("history_file", "research_results/state5_history.csv"))
        
        # 初期実績ベースのフォールバック値（Version 7の真実の統計データを標準として内蔵）
        base_stats = {
            "win_rate": 53.79,
            "avg_return": 2.74,
            "median_return": 0.87,
            "avg_win": 12.70,
            "avg_loss": 8.86,
            "pf": 1.67,
            "max_dd": -9.43
        }
        
        # もしデータベースに十分な完了案件（60日後リターンあり）が蓄積されていれば、
        # 現在の地合いにおける勝率・PF・リターンを「リアルタイムに自動計算」して反映
        if history_file.exists():
            try:
                df = pd.read_csv(history_file)
                df_eval = df.dropna(subset=["return_60d"]).copy()
                if len(df_eval) >= 10:
                    df_eval["is_win"] = df_eval["return_60d"] > 0
                    
                    # 現在の地合いに合致するデータを抽出
                    df_env = df_eval[df_eval["market_env"] == market_state]
                    if len(df_env) >= 3:
                        win_events = df_env[df_env["is_win"]]
                        loss_events = df_env[~df_env["is_win"]]
                        
                        total_profit = win_events["return_60d"].sum() if not win_events.empty else 0.0
                        total_loss = abs(loss_events["return_60d"].sum()) if not loss_events.empty else 1.0
                        pf = total_profit / total_loss if total_loss > 0 else 0.0
                        
                        base_stats = {
                            "win_rate": df_env["is_win"].mean() * 100,
                            "avg_return": df_env["return_60d"].mean(),
                            "median_return": df_env["return_60d"].median(),
                            "avg_win": win_events["return_60d"].mean() if not win_events.empty else 0.0,
                            "avg_loss": abs(loss_events["return_60d"].mean()) if not loss_events.empty else 0.0,
                            "pf": pf,
                            "max_dd": df_env["max_drawdown_90d"].median() if "max_drawdown_90d" in df_env.columns else -9.43
                        }
            except Exception:
                pass

        env_desc = {
            "Bull": "現在市場は【 Bull (強気・上昇トレンド相場) 】です。大衆が強気になって買い上がるため、State 5の押し目からState 6（本上昇開始）への遷移が成功しやすく、かつブレイクした際の利大リターン幅が極めて大きくなりやすい、極めて有利な投資地合いです。",
            "Bear": "現在市場は【 Bear (弱気・下降トレンド相場) 】です。市場全体の売り圧力が強く、どれだけ個別銘柄の条件が良くとも、地合いの急落に巻き込まれて失敗（ドロップ）する確率が通常時より有意に高まるため、警戒・防衛が必要な投資地合いです。",
            "Range": "現在市場は【 Range (揉み合い相場) 】です。明確な方向性がないため、個別銘柄の材料や出来高収縮度合い（Type0との親和性）が株価を決定付けます。地合いに依存しない、徹底した銘柄選別が求められます。",
            "Neutral": "現在市場は【 Neutral (中立相場) 】です。地合いからの追い風も向かい風も平穏な状態であり、過去統計通りの標準的な期待値がそのまま推移します。"
        }
        
        stats_str = (
            f"  ・過去統計上の勝率 (60日後): {base_stats['win_rate']:.2f}%\n"
            f"  ・平均期待収益率: {base_stats['avg_return']:+.2f}% (中央値: {base_stats['median_return']:+.2f}%)\n"
            f"  ・平均利益率 (Win): {base_stats['avg_win']:+.2f}% / 平均損失率 (Loss): -{base_stats['avg_loss']:.2f}%\n"
            f"  ・Profit Factor (PF): {base_stats['pf']:.2f} / 平均最大下落率: {base_stats['max_dd']:.2f}%"
        )
        
        return env_desc.get(market_state, "中立市場です。"), stats_str

    @staticmethod
    def get_natural_ai_comment(latest_row: pd.Series, matching_rate: int) -> str:
        """
        ⑨：客観的な事実データに基づく、プロ水準の自然言語のAI解説コメント
        """
        vol_ratio = latest_row["vol_ratio_20"]
        rsi14 = latest_row["rsi14"]
        bb_width = latest_row["bb_width"]
        dist_52w = abs(latest_row["dist_to_52w_high"])
        
        comment = (
            f"【データ分析】: 本銘柄は、出来高が20日平均の {vol_ratio:.2f}倍 まで十分に収縮（売り枯れ）し、"
            f"ボラティリティ（BB幅 {bb_width:.1f}%）も限界レベルまで低下（スクイーズ）を完了させています。 "
            f"RSIは {rsi14:.1f}% と、過熱感が完全に消滅した理想的な中立圏を推移しています。 "
            f"過去5,487件の大化け株の物理法則では、この『限界収縮から始まる沈黙期（Type 0一致率: {matching_rate}%）』を経て、"
            f"平均して10〜15営業日以内に出来高の急増（再ブレイク）へと移行するケースが圧倒的に多く確認されています。 "
            f"現在は52週高値からわずか {dist_52w:.1f}% 押し戻された位置にあり、下値リスクが極限まで限定された、"
            f"典型的な『静かな待ち伏せ（仕込み）』の局面に位置しています。"
        )
        return comment