# early_history_logger.py (Version 1.0 - Early Watch Logger)
import pandas as pd
from pathlib import Path

class EarlyHistoryLogger:
    """
    Early Watch専用の学習データベース（履歴）自動記録クラス
    """
    @staticmethod
    def log_early_candidates(candidates: list[dict], date_str: str, config: dict):
        history_file = Path(config.get("research", {}).get("early_history_file", "research_results/early_history.csv"))
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 既存データベースのロード（なければ新規作成）
        if history_file.exists():
            df_hist = pd.read_csv(history_file, encoding="utf-8-sig")
        else:
            df_hist = pd.DataFrame(columns=[
                "date", "ticker", "name", "congestion_width", "congestion_duration", "rsi14", "bb_width", "vol_ratio", "dist_52w",
                "status", "days_held", "max_high_120d", "max_drawdown_120d", "last_state",
                "reached_state3_days", "reached_state4_days", "reached_state6_days", "failed_reason"
            ])
            
        new_rows = []
        for c in candidates:
            # 同一日の重複登録を完全に回避
            is_dup = not df_hist[(df_hist["date"] == date_str) & (df_hist["ticker"] == c["ticker"])].empty
            if is_dup:
                continue
                
            new_rows.append({
                "date": date_str,
                "ticker": c["ticker"],
                "name": c["name"],
                "congestion_width": c["congestion_width"],
                "congestion_duration": c["congestion_duration"],
                "rsi14": c["rsi14"],
                "bb_width": c["bb_width"],
                "vol_ratio": c["vol_ratio"],
                "dist_52w": c["dist_52w"],
                "status": "tracking",  # 追跡中
                "days_held": 0,
                "max_high_120d": 0.0,
                "max_drawdown_120d": 0.0,
                "last_state": "State 1",
                "reached_state3_days": None,
                "reached_state4_days": None,
                "reached_state6_days": None,
                "failed_reason": None
            })
            
        if new_rows:
            df_new = pd.DataFrame(new_rows)
            df_combined = pd.concat([df_hist, df_new], ignore_index=True)
            df_combined.to_csv(history_file, index=False, encoding="utf-8-sig")
            print(f"  [Early台帳記録] 本日の予備軍 {len(new_rows)} 件の『学習データ』を {history_file.name} に保存しました。")