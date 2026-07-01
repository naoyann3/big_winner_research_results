# state5_history_logger.py
import pandas as pd
from pathlib import Path

class State5HistoryLogger:
    @staticmethod
    def log_candidates(candidates: list[dict], date_str: str, market_env: dict, config: dict):
        history_file = Path(config.get("research", {}).get("history_file", "research_results/state5_history.csv"))
        history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 既存データベースのロード（なければ新規作成）
        if history_file.exists():
            df_hist = pd.read_csv(history_file, encoding="utf-8-sig")
        else:
            df_hist = pd.DataFrame(columns=[
                "date", "ticker", "name", "state", "score", "close", "vol_ratio", "bb_width", "rsi14", "ma75_dev",
                "dist_to_52w_high", "dist_to_52w_low", "ma25_slope", "atr_ratio",
                "market_env", "status", "days_held",
                "return_30d", "return_60d", "return_90d", "max_high_90d", "max_drawdown_90d"
            ])
            
        new_rows = []
        for c in candidates:
            # 重複の防止 (同じ日、同じ銘柄の重複登録を回避)
            is_dup = not df_hist[(df_hist["date"] == date_str) & (df_hist["ticker"] == c["ticker"])].empty
            if is_dup:
                continue
                
            new_rows.append({
                "date": date_str,
                "ticker": c["ticker"],
                "name": c["name"],
                "state": f"State {c['state']}",
                "score": c["score"],
                "close": c["close"],
                "vol_ratio": c["vol_ratio"],
                "bb_width": c["bb_width"],
                "rsi14": c["rsi14"],
                "ma75_dev": c["ma75_dev"],
                "dist_to_52w_high": c["dist_to_52w_high"],
                "dist_to_52w_low": c["dist_to_52w_low"],
                "ma25_slope": c["ma25_slope"],
                "atr_ratio": c["atr_ratio"],
                "market_env": market_env["market_state_topix"],
                "status": "tracking",
                "days_held": 0,
                "return_30d": None,
                "return_60d": None,
                "return_90d": None,
                "max_high_90d": None,
                "max_drawdown_90d": None
            })
            
        if new_rows:
            df_new = pd.DataFrame(new_rows)
            df_combined = pd.concat([df_hist, df_new], ignore_index=True)
            df_combined.to_csv(history_file, index=False, encoding="utf-8-sig")
            print(f"  [データ蓄積完了] 本日のState 5 候補 {len(new_rows)} 件の『教師データ』を {history_file.name} に保存しました。")