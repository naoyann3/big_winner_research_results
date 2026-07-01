# performance_tracker.py
import pandas as pd
from pathlib import Path
import numpy as np

class PerformanceTracker:
    @staticmethod
    def track_and_score_history(config: dict):
        history_file = Path(config.get("research", {}).get("history_file", "research_results/state5_history.csv"))
        prices_dir = Path("data_cache/prices")
        
        if not history_file.exists():
            return
            
        df_hist = pd.read_csv(history_file, encoding="utf-8-sig")
        if df_hist.empty:
            return
            
        tracking_mask = df_hist["status"] == "tracking"
        updated_count = 0
        
        for idx in df_hist[tracking_mask].index:
            ticker = df_hist.at[idx, "ticker"]
            entry_date_str = df_hist.at[idx, "date"]
            entry_close = float(df_hist.at[idx, "close"])
            
            price_path = prices_dir / f"{ticker}.csv"
            if not price_path.exists():
                continue
                
            try:
                d = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()
                d_after = d.loc[entry_date_str:]
                if d_after.empty:
                    continue
                    
                days_held = len(d_after) - 1
                df_hist.at[idx, "days_held"] = days_held
                
                # 30, 60, 90営業日時点でのリターンを自動計算して採点
                for period in [30, 60, 90]:
                    if days_held >= period:
                        if len(d_after) > period:
                            p_close = float(d_after.iloc[period]["Close"])
                            df_hist.at[idx, f"return_{period}d"] = round((p_close - entry_close) / entry_close * 100, 2)
                
                # 90営業日以内の最高値・最大ドローダウンを自動計算して採点
                window_90 = d_after.head(91)
                if len(window_90) > 1:
                    max_high = float(window_90["High"].max())
                    df_hist.at[idx, "max_high_90d"] = round((max_high - entry_close) / entry_close * 100, 2)
                    
                    cum_max = window_90["High"].cummax()
                    drawdowns = (window_90["Low"] - cum_max) / cum_max * 100
                    df_hist.at[idx, "max_drawdown_90d"] = round(float(drawdowns.min()), 2)
                
                # 90営業日が経過したらステータスを「完了」にする
                if days_held >= 90:
                    df_hist.at[idx, "status"] = "completed"
                    
                updated_count += 1
            except Exception:
                continue
                
        if updated_count > 0:
            df_hist.to_csv(history_file, index=False, encoding="utf-8-sig")
            print(f"  [自動成績採点完了] {updated_count} 件の過去シグナルについて、最新株価から成績を自動アップデートしました。")