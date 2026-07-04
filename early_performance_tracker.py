# early_performance_tracker.py (Version 1.0 - Early Performance & Evolution Tracker)
import pandas as pd
from pathlib import Path
import numpy as np

class EarlyPerformanceTracker:
    """
    Early Watch専用の自動成績追跡・進化・失敗イベント自動検知クラス
    """
    @staticmethod
    def track_and_detect_evolutions(config: dict) -> list[str]:
        history_file = Path(config.get("research", {}).get("early_history_file", "research_results/early_history.csv"))
        prices_dir = Path("data_cache/prices")
        
        if not history_file.exists():
            return []
            
        df_hist = pd.read_csv(history_file, encoding="utf-8-sig")
        if df_hist.empty:
            return []
            
        tracking_mask = df_hist["status"] == "tracking"
        updated_count = 0
        evolution_alerts = []  # 本日発生した進化・失敗アラートを記録するリスト
        
        # 状態遷移を再判定するためのインポート
        from state5_monitoring_system import MarketStateEngine
        
        for idx in df_hist[tracking_mask].index:
            ticker = df_hist.at[idx, "ticker"]
            name = df_hist.at[idx, "name"]
            entry_date_str = df_hist.at[idx, "date"]
            entry_close = float(df_hist.at[idx, "close"]) if "close" in df_hist.columns and not pd.isna(df_hist.at[idx, "close"]) else None
            
            price_path = prices_dir / f"{ticker}.csv"
            if not price_path.exists():
                continue
                
            try:
                d_raw = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()
                d_ind = MarketStateEngine.calculate_indicators(d_raw)
                d_sim = MarketStateEngine.simulate_state_machine(d_ind)
                
                # 進入日以降のデータ
                d_after = d_sim.loc[entry_date_str:]
                if d_after.empty:
                    continue
                    
                days_held = len(d_after) - 1
                df_hist.at[idx, "days_held"] = days_held
                
                # 今日の最新データと前日のデータ
                latest_row = d_after.iloc[-1]
                prev_row = d_after.iloc[-2] if len(d_after) > 1 else latest_row
                
                current_state = int(latest_row["current_state"])
                prev_state = int(prev_row["current_state"])
                
                close = float(latest_row["Close"])
                ma25 = float(latest_row["ma25"])
                ma75 = float(latest_row["ma75"])
                ma200 = float(latest_row["ma200"])
                vol_ratio = float(latest_row["vol_ratio_20"])
                
                # 進入時の価格が不明な場合は、進入日のCloseを代入
                if entry_close is None:
                    entry_close = float(d_after.iloc[0]["Close"])
                    df_hist.at[idx, "close"] = entry_close
                
                # ① 【進化イベント検知】: State 3（狼煙）に本日到達
                if current_state >= 3 and prev_state < 3 and pd.isna(df_hist.at[idx, "reached_state3_days"]):
                    df_hist.at[idx, "reached_state3_days"] = days_held
                    evolution_alerts.append(
                        f"🚀 **【進化：State 3到達】** 登録から {days_held}日目: **{name} ({ticker})** が出来高 **{vol_ratio:.1f}倍** を記録し、水面下から大口の『狼煙（State 3）』へ進化しました！"
                    )
                    
                # ② 【進化イベント検知】: State 4（第一波・大爆発）に本日到達
                if current_state >= 4 and prev_state < 4 and pd.isna(df_hist.at[idx, "reached_state4_days"]):
                    df_hist.at[idx, "reached_state4_days"] = days_held
                    evolution_alerts.append(
                        f"🔥 **【進化：State 4到達】** 登録から {days_held}日目: **{name} ({ticker})** が出来高 **{vol_ratio:.1f}倍** の大爆発を起こし、『第一波（State 4：True Day 0）』へ昇格しました！"
                    )
                    
                # ③ 【進化イベント検知】: State 5（本番用Gold Watch対象）に本日到達
                if current_state == 5 and prev_state < 5 and pd.isna(df_hist.at[idx, "reached_state6_days"]):
                    df_hist.at[idx, "reached_state6_days"] = days_held
                    evolution_alerts.append(
                        f"👑 **【王冠：Gold Watch昇格】** 登録から {days_held}日目: **{name} ({ticker})** が売り枯れの極限を完了し、本番の『Gold Watch仕込み候補（State 5）』へ昇格しました！"
                    )
                    
                # ④ 【進化イベント検知】: 本日、パーフェクトオーダーが完成
                if (ma25 > ma75 > ma200) and not (prev_row["ma25"] > prev_row["ma75"] > prev_row["ma200"]):
                    evolution_alerts.append(
                        f"📈 **【進化：PO完成】** 登録から {days_held}日目: **{name} ({ticker})** の移動平均線が完全なる『上昇パーフェクトオーダー（25 > 75 > 200）』を本日完成させました！"
                    )

                # ⑤ 【失敗イベント検知】: 75日線、または200日線の下割れ
                if close < ma75 and prev_row["Close"] >= prev_row["ma75"]:
                    evolution_alerts.append(
                        f"💦 **【警告：75日線割れ】** 登録から {days_held}日目: **{name} ({ticker})** が下値支持帯である75日移動平均線を本日下方向に割り込みました。黄色信号です。"
                    )
                    
                if close < ma200 and prev_row["Close"] >= prev_row["ma200"]:
                    df_hist.at[idx, "status"] = "completed"
                    df_hist.at[idx, "failed_reason"] = "MA200_Break"
                    evolution_alerts.append(
                        f"❌ **【脱落：長期線割れ】** 登録から {days_held}日目: **{name} ({ticker})** が長期的な防衛線である200日移動平均線を完全に下回り、トレンドが崩壊したため『追跡を終了（脱落）』します。"
                    )

                # ⑥ 【失敗イベント検知】: 登録時の価格から12%以上下落（損切り・ドロップ）
                ret_pct = (close - entry_close) / entry_close * 100
                if ret_pct <= -12.0:
                    df_hist.at[idx, "status"] = "completed"
                    df_hist.at[idx, "failed_reason"] = "Max_Loss_Limit"
                    evolution_alerts.append(
                        f"❌ **【脱落：12%損切り基準到達】** 登録から {days_held}日目: **{name} ({ticker})** が登録時の価格から **{ret_pct:.1f}%** 下落したため、損切り基準に達し『追跡を終了（脱落）』します。"
                    )

                # 最高上昇率・最大下落率の毎日アップデート
                max_high = float(d_after["High"].max())
                min_low = float(d_after["Low"].min())
                
                df_hist.at[idx, "max_high_120d"] = round((max_high - entry_close) / entry_close * 100, 2)
                df_hist.at[idx, "max_drawdown_120d"] = round((min_low - entry_close) / entry_close * 100, 2)
                df_hist.at[idx, "last_state"] = f"State {current_state}"

                # 120営業日（半年）が経過したら、自動で追跡を正常終了させる
                if days_held >= 120:
                    df_hist.at[idx, "status"] = "completed"
                    df_hist.at[idx, "failed_reason"] = "Timeout_120d"
                    
                updated_count += 1
            except Exception as e:
                continue
                
        if updated_count > 0:
            df_hist.to_csv(history_file, index=False, encoding="utf-8-sig")
            print(f"  [Early追跡完了] {updated_count} 件の過去の予備軍を追跡・採点しました。本日発生したアラートは 【 {len(evolution_alerts)} 件 】 です。")
            
        return evolution_alerts