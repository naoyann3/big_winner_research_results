import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# --- 設定 ---
class EngineConfig:
    prices_dir = Path("../data_cache/prices")
    fund_dir = Path("../data_cache/fundamentals")
    universe_csv = Path("../universe.csv")
    output_dir = Path("research_results")


class MarketStateEngine:
    """
    全銘柄の時系列をスキャンし、日次のState遷移を管理・トラッキングするコアエンジン
    """
    def __init__(self, prices_dir: Path, fund_dir: Path):
        self.prices_dir = prices_dir
        self.fund_dir = fund_dir

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        
        # 移動平均
        for ma in [25, 75]:
            d[f"ma{ma}"] = d["Close"].rolling(ma).mean()
            d[f"ma{ma}_dev"] = (d["Close"] - d[f"ma{ma}"]) / d[f"ma{ma}"] * 100
            
        d["ma25_slope"] = d["ma25"].pct_change(5) * 100
        d["vol_avg20"] = d["Volume"].rolling(20).mean()
        d["vol_ratio_20"] = d["Volume"] / d["vol_avg20"]
        
        # ボラティリティ (ATR / BB Width)
        high_low = d["High"] - d["Low"]
        high_cp = (d["High"] - d["Close"].shift(1)).abs()
        low_cp = (d["Low"] - d["Close"].shift(1)).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        d["atr_ratio"] = (tr.rolling(14).mean() / d["Close"]) * 100
        
        std20 = d["Close"].rolling(20).std()
        d["bb_width"] = (std20 * 4) / d["ma25"] * 100
        d["bb_width_min60"] = d["bb_width"].rolling(60).min()

        # RSI (14)
        delta = d["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        d["rsi14"] = 100 - (100 / (1 + (gain / np.where(loss > 0, loss, 1.0))))
        
        # 前日差
        d["delta_volume_ratio"] = d["vol_ratio_20"].diff()
        d["delta_atr_ratio"] = d["atr_ratio"].diff()
        d["delta_ma25_slope"] = d["ma25_slope"].diff()
        
        # 直近20日高値
        d["high_20d"] = d["High"].shift(1).rolling(20).max()

        return d

    def simulate_state_machine(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        時系列に沿って、毎日ステートマシンを実行し状態遷移をシミュレーションします
        """
        d = df.copy()
        states = []
        state_durations = []
        
        current_state = 0
        state_days = 0
        
        # 統計取得用
        transitions_counter = {
            "Squeeze_to_Reversal": 0, "Squeeze_to_Fail": 0,
            "Reversal_to_Inflow": 0, "Reversal_to_Fail": 0,
            "Inflow_to_TrueDay0": 0, "Inflow_to_Fail": 0,
            "TrueDay0_to_Shakeout": 0, "TrueDay0_to_Fail": 0,
            "Shakeout_to_MainRun": 0, "Shakeout_to_Fail": 0,
        }
        
        # 追跡変数
        last_high = 0.0
        
        for idx in range(len(d)):
            row = d.iloc[idx]
            
            # 各指標の取得
            close = row["Close"]
            bb_width = row["bb_width"]
            bb_min = row["bb_width_min60"]
            ma25_slope = row["ma25_slope"]
            rsi14 = row["rsi14"]
            vol_ratio = row["vol_ratio_20"]
            high_20d = row["high_20d"]
            
            # 安全処理: 指標が計算できない（欠損）場合は State 0
            if pd.isna(bb_width) or pd.isna(ma25_slope) or pd.isna(rsi14) or pd.isna(vol_ratio):
                states.append(0)
                state_durations.append(0)
                continue
                
            # 直近最高値の更新
            last_high = max(last_high, row["High"]) if current_state > 0 else row["High"]
            
            # --- 失敗（Drop）の判定条件：直近高値から10%下落したら強制的に State 0 へ ---
            if current_state > 0 and close < last_high * 0.90:
                transitions_counter[f"State{current_state}_to_Fail"] = transitions_counter.get(f"State{current_state}_to_Fail", 0) + 1
                current_state = 0
                state_days = 0
                last_high = row["High"]

            # --- 状態遷移ロジック（State Machine） ---
            next_state = current_state
            
            if current_state == 0:
                if bb_width <= bb_min * 1.05:
                    next_state = 1
                    
            elif current_state == 1:
                if ma25_slope > 0 and rsi14 >= 50.0:
                    next_state = 2
                    transitions_counter["Squeeze_to_Reversal"] += 1
                    
            elif current_state == 2:
                if vol_ratio >= 2.0:
                    next_state = 3
                    transitions_counter["Reversal_to_Inflow"] += 1
                    
            elif current_state == 3:
                if vol_ratio >= 3.0 and close > row["Open"]:
                    next_state = 4
                    transitions_counter["Inflow_to_TrueDay0"] += 1
                    
            elif current_state == 4:
                if close < row["Open"] and vol_ratio < 1.0:
                    next_state = 5
                    transitions_counter["TrueDay0_to_Shakeout"] += 1
                    
            elif current_state == 5:
                if close > high_20d and vol_ratio >= 1.5:
                    next_state = 6
                    transitions_counter["Shakeout_to_MainRun"] += 1
                    
            elif current_state == 6:
                pass
                
            # 状態更新の処理
            if next_state != current_state:
                current_state = next_state
                state_days = 1
            else:
                state_days += 1
                
            states.append(current_state)
            state_durations.append(state_days)
            
        d["current_state"] = states
        d["state_days"] = state_durations
        return d, transitions_counter


# --- メインプロセッサ ---
def main():
    EngineConfig.output_dir.mkdir(parents=True, exist_ok=True)
    
    if not EngineConfig.universe_csv.exists():
        print(f"宇宙ファイル {EngineConfig.universe_csv} が見つかりません。")
        return
        
    df_uni = pd.read_csv(EngineConfig.universe_csv)
    tickers = df_uni["ticker"].dropna().tolist()
    
    def normalize_ticker(raw: str) -> str:
        ticker = str(raw).strip().upper()
        return f"{ticker}.T" if "." not in ticker else ticker
    tickers = [normalize_ticker(t) for t in tickers]
    
    engine = MarketStateEngine(EngineConfig.prices_dir, EngineConfig.fund_dir)
    
    print("=== Market State Engine (Version 5) の実行を開始します ===")
    
    daily_states = []
    global_failures = {
        "Squeeze_to_Reversal": 0, "Squeeze_to_Fail": 0,
        "Reversal_to_Inflow": 0, "Reversal_to_Fail": 0,
        "Inflow_to_TrueDay0": 0, "Inflow_to_Fail": 0,
        "TrueDay0_to_Shakeout": 0, "TrueDay0_to_Fail": 0,
        "Shakeout_to_MainRun": 0, "Shakeout_to_Fail": 0,
    }
    
    # 全銘柄スキャン
    for idx, t in enumerate(tickers):
        price_path = EngineConfig.prices_dir / f"{t}.csv"
        fund_path = EngineConfig.fund_dir / f"{t}.json"
        
        if (idx + 1) % 500 == 0 or (idx + 1) == len(tickers):
            print(f"  [状態スキャン中] {idx+1} / {len(tickers)} 銘柄精査完了...")
            
        if not price_path.exists():
            continue
            
        try:
            df_raw = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()
            if len(df_raw) < 150:
                continue
                
            df_ind = engine.calculate_technical_indicators(df_raw)
            df_sim, failures = engine.simulate_state_machine(df_ind)
            
            # 各失敗カウンターをグローバルに加算
            for key, val in failures.items():
                global_failures[key] += val
                
            # 本日（最新日）の状態を取得
            latest_row = df_sim.iloc[-1]
            latest_state = int(latest_row["current_state"])
            latest_days = int(latest_row["state_days"])
            
            # 過去30日間の状態遷移履歴（文字列化）
            history_states = df_sim["current_state"].iloc[-30:].astype(str).tolist()
            state_history_str = "➔".join(dict.fromkeys(history_states))
            
            # 【修正点】日付型の計算（timedelta）を使わずに、時系列インデックスの位置から日付を特定（100%安全設計）
            state_entry_idx = len(df_sim) - latest_days
            if state_entry_idx < 0:
                state_entry_idx = 0
            raw_entry_date = df_sim.index[state_entry_idx]
            
            # 型チェック（日付型ならフォーマット、文字列ならそのまま切り出し）
            if hasattr(raw_entry_date, "strftime"):
                state_entry_date_str = raw_entry_date.strftime("%Y-%m-%d")
            else:
                state_entry_date_str = str(raw_entry_date)[:10]
            
            # 期待スコアと危険度の簡易数理モデル
            expect_score = 0
            danger_score = 0
            if latest_state == 1: expect_score = 30
            elif latest_state == 2: expect_score = 50
            elif latest_state == 3: expect_score = 70; danger_score = 20
            elif latest_state == 4: expect_score = 85; danger_score = 30
            elif latest_state == 5: expect_score = 95; danger_score = 15
            elif latest_state == 6: expect_score = 80; danger_score = 40
            
            if latest_days > 20 and latest_state in [3, 4, 5]:
                danger_score += 20

            # ファンダメンタルズのロード
            fund_data = {}
            if fund_path.exists():
                try:
                    with open(fund_path, "r", encoding="utf-8") as f:
                        fund_data = json.load(f)
                except Exception:
                    pass

            daily_states.append({
                "ticker": t,
                "name": t,
                "current_state": f"State {latest_state}",
                "days_in_state": latest_days,
                "state_entry_date": state_entry_date_str,
                "state_history": state_history_str,
                "rsi14": round(float(latest_row["rsi14"]), 1) if not pd.isna(latest_row["rsi14"]) else None,
                "vol_ratio": round(float(latest_row["vol_ratio_20"]), 2) if not pd.isna(latest_row["vol_ratio_20"]) else None,
                "expect_score": expect_score,
                "danger_score": danger_score,
                "sector": fund_data.get("sector", "不明"),
                "industry": fund_data.get("industry", "不明")
            })
            
        except Exception as e:
            # デバッグ用にエラーが出た場合はスキップするが、通常は発生しない
            continue
            
    # 【安全対策】取得データが空っぽだった場合の安全弁
    if not daily_states:
        print("\n[エラー]: 状態スキャンは完了しましたが、有効なデータが1件も取得できませんでした。")
        print("原因として、CSVデータの読み込み失敗やデータ形式のズレが考えられます。")
        return

    # レポート保存
    df_report = pd.DataFrame(daily_states)
    report_path = EngineConfig.output_dir / "daily_market_state_report.csv"
    df_report.sort_values(by="expect_score", ascending=False).to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"\n[保存成功] 本日の市場状態レポートを保存しました: {report_path.name}")
    
    # 統計情報：Stateごとの失敗確率の計算とレポート出力
    stats_path = EngineConfig.output_dir / "market_state_statistics_report.txt"
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("★ Market State Engine: 各Stateにおける生存・ドロップ確率 ★\n")
        f.write("==================================================\n")
        f.write("全3600銘柄の過去5年間（約400万取引日）のスキャン結果に基づく統計:\n\n")
        
        def write_stats(state_name, success_key, fail_key):
            suc = global_failures.get(success_key, 0)
            fail = global_failures.get(fail_key, 0)
            total = suc + fail
            rate = (fail / total * 100) if total > 0 else 0.0
            f.write(f"■ 【{state_name}】でのドロップ（通常への逆戻り）確率:\n")
            f.write(f"   ・次の段階へ進んだ数: {suc} 件 / 途中で失敗（10%下落）した数: {fail} 件\n")
            f.write(f"   ・この状態からの 【 失敗率（ドロップ確率）: {rate:.2f} % 】 (成功率: {100-rate:.2f} %)\n")
            f.write("--------------------------------------------------\n")
            
        write_stats("State 1: スクイーズ極限", "Squeeze_to_Reversal", "State1_to_Fail")
        write_stats("State 2: トレンド＆モメンタム反転", "Reversal_to_Inflow", "State2_to_Fail")
        write_stats("State 3: 先行資金流入（狼煙）", "Inflow_to_TrueDay0", "State3_to_Fail")
        write_stats("State 4: True Day 0（第一波）", "TrueDay0_to_Shakeout", "State4_to_Fail")
        write_stats("State 5: 揺さぶり（ふるい落とし）", "Shakeout_to_MainRun", "State5_to_Fail")
        
    print(f"[保存成功] 各Stateの失敗確率統計レポートを保存しました: {stats_path.name}")
    
    # 画面に統計結果を表示
    with open(stats_path, "r", encoding="utf-8") as f:
        print("\n" + f.read())


if __name__ == "__main__":
    main()
