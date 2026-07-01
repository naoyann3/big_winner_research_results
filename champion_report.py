# champion_report.py
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime

class ChampionReportGenerator:
    @staticmethod
    def generate_report(config: dict):
        history_file = Path(config.get("research", {}).get("history_file", "research_results/state5_history.csv"))
        report_file = Path(config.get("research", {}).get("report_file", "research_results/champion_report.md"))
        
        if not history_file.exists():
            return
            
        df = pd.read_csv(history_file)
        if df.empty:
            return
            
        # 60日後のリターンが存在する「評価可能な（十分な追跡期間を経た）案件」で統計を計算
        df_eval = df.dropna(subset=["return_60d"]).copy()
        
        total_eval = len(df_eval)
        if total_eval == 0:
            # まだ60営業日経過したシグナルがない場合のプレースホルダー
            with open(report_file, "w", encoding="utf-8") as f:
                f.write("# Champion Report (成績評価レポート)\n")
                f.write("現在、60営業日以上経過した評価対象案件がありません。データ蓄積および日次追跡を継続してください。\n")
            return
            
        df_eval["is_win"] = df_eval["return_60d"] > 0
        win_rate = df_eval["is_win"].mean() * 100
        
        win_events = df_eval[df_eval["is_win"]]
        loss_events = df_eval[~df_eval["is_win"]]
        
        avg_win = win_events["return_60d"].mean() if not win_events.empty else 0.0
        avg_loss = abs(loss_events["return_60d"].mean()) if not loss_events.empty else 0.0
        
        total_profit = win_events["return_60d"].sum() if not win_events.empty else 0.0
        total_loss = abs(loss_events["return_60d"].sum()) if not loss_events.empty else 1.0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
        
        expectancy = (win_rate/100 * avg_win) - ((1 - win_rate/100) * avg_loss)
        avg_max_high = df_eval["max_high_90d"].mean() if "max_high_90d" in df_eval.columns else 0.0
        avg_max_dd = df_eval["max_drawdown_90d"].mean() if "max_drawdown_90d" in df_eval.columns else 0.0
        
        # --- 【Version 7新設】：市場環境（market_env）別の勝率・期待値クロス集計 ---
        env_summary_rows = []
        if "market_env" in df_eval.columns:
            for env, grp in df_eval.groupby("market_env"):
                env_win_rate = grp["is_win"].mean() * 100
                env_avg_ret = grp["return_60d"].mean()
                env_avg_dd = grp["max_drawdown_90d"].mean() if "max_drawdown_90d" in grp.columns else 0.0
                env_summary_rows.append(
                    f"| {env:10s} | {len(grp):5d} | {env_win_rate:6.2f}% | {env_avg_ret:+7.2f}% | {env_avg_dd:6.2f}% |"
                )
        env_table_str = "\n".join(env_summary_rows) if env_summary_rows else "| (データなし) |"

        # レポートの作成（Markdown形式で美しく自動生成）
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# 【Champion Report】市場状態・投資成績評価レポート\n\n")
            f.write(f"最終生成日: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"評価完了案件総数: {total_eval} 件 (累積台帳総数: {len(df)} 件)\n\n")
            
            f.write("## 1. 総合投資成績 (60営業日後基準)\n")
            f.write("--------------------------------------------------\n")
            f.write(f"*   **勝率 (60日後リターン > 0)** : **{win_rate:.2f} %**\n")
            f.write(f"*   **平均利益率 (Win)**          : **{avg_win:+.2f} %**\n")
            f.write(f"*   **平均損失率 (Loss)**         : **-{avg_loss:.2f} %**\n")
            f.write(f"*   **Profit Factor (PF)**        : **{profit_factor:.2f}**\n")
            f.write(f"*   **統計的期待値 (Expectancy)**  : **{expectancy:+.2f} %** (1取引あたりの期待期待収益率)\n")
            f.write(f"*   **90日以内の平均最大上昇率**   : **{avg_max_high:+.2f} %**\n")
            f.write(f"*   **90日以内の平均最大下落率**   : **{avg_max_dd:.2f} %**\n")
            f.write("--------------------------------------------------\n\n")
            
            f.write("## 2. Market Environment (相場環境別の勝率・パフォーマンス)\n")
            f.write("市場全体の地合い（TOPIXのトレンド）によって、期待値がどのように変化するかを分類したクオンツ分析表です。\n\n")
            f.write("| 市場環境   | 案件数 | 勝率    | 平均リターン | 平均最大下落率 |\n")
            f.write("| :---       | :---   | :---    | :---        | :---           |\n")
            f.write(env_table_str + "\n\n")
            
            f.write("## 3. クオンツ改善へのアドバイス\n")
            f.write("*   **地合い別戦略の検討**: Bull（強気市場）とBear（弱気市場）で勝率に有意な差がある場合、Bear時には自動的に仕込み件数を減らす（または投資資金を抑える）等の動的ベータ制限が有効になります。\n")
            f.write("*   **最大下落率の許容**: 平均最大下落率（Max DD）を基準に、最も効率的なロスカットライン（例：平均DDの1.2倍の位置に置くなど）の最適設計に活用してください。\n")
            
        print(f"  [レポート自動作成完了] 実績評価レポートを保存しました: {report_file.name}")