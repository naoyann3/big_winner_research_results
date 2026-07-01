import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 機械学習ライブラリのインポート
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --- 設定 ---
class AdvancedConfig:
    prices_dir = Path("../data_cache/prices")
    fund_dir = Path("../data_cache/fundamentals")
    universe_csv = Path("../universe.csv")
    output_dir = Path("research_results")
    
    # 将来リターンの追跡日数
    forward_periods = [5, 10, 20, 40, 60, 120]


class State5AdvancedAnalyzer:
    def __init__(self, prices_dir: Path):
        self.prices_dir = prices_dir

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["ma25"] = d["Close"].rolling(25).mean()
        d["ma25_slope"] = d["ma25"].pct_change(5) * 100
        d["vol_avg20"] = d["Volume"].rolling(20).mean()
        d["vol_ratio_20"] = d["Volume"] / d["vol_avg20"]
        
        # ボラティリティ
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
        
        d["high_20d"] = d["High"].shift(1).rolling(20).max()
        return d

    def simulate_states_and_collect_events(self, df: pd.DataFrame, ticker: str) -> list[dict]:
        """
        時系列に沿って状態遷移をシミュレートし、State 5に進入した『瞬間（初日）』のイベントを全サンプリング
        """
        states = []
        current_state = 0
        last_high = 0.0
        
        events = []
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            close = row["Close"]
            bb_width = row["bb_width"]
            bb_min = row["bb_width_min60"]
            ma25_slope = row["ma25_slope"]
            rsi14 = row["rsi14"]
            vol_ratio = row["vol_ratio_20"]
            high_20d = row["high_20d"]
            
            if pd.isna(bb_width) or pd.isna(ma25_slope) or pd.isna(rsi14) or pd.isna(vol_ratio):
                states.append(0)
                continue
                
            last_high = max(last_high, row["High"]) if current_state > 0 else row["High"]
            
            # 失敗判定
            if current_state > 0 and close < last_high * 0.90:
                current_state = 0
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
                
            # 【State 5 進入イベントの記録】
            if next_state == 5 and current_state != 5:
                events.append({
                    "ticker": ticker,
                    "date": df.index[idx],
                    "idx_pos": idx,
                    "entry_close": close,
                    # クラスタリング用の特徴量
                    "vol_ratio": vol_ratio,
                    "rsi14": rsi14,
                    "atr_ratio": row["atr_ratio"],
                    "bb_width": bb_width,
                    "ma25_slope": ma25_slope
                })
                
            if next_state != current_state:
                current_state = next_state
            states.append(current_state)
            
        return events


# --- メインプロセッサ ---
def main():
    AdvancedConfig.output_dir.mkdir(parents=True, exist_ok=True)
    
    if not AdvancedConfig.universe_csv.exists():
        print("universe.csv が見つかりません。")
        return
        
    df_uni = pd.read_csv(AdvancedConfig.universe_csv)
    tickers = df_uni["ticker"].dropna().tolist()
    
    def normalize_ticker(raw: str) -> str:
        ticker = str(raw).strip().upper()
        return f"{ticker}.T" if "." not in ticker else ticker
    tickers = [normalize_ticker(t) for t in tickers]
    
    analyzer = State5AdvancedAnalyzer(AdvancedConfig.prices_dir)
    
    print("=== Step 1: 全3600銘柄から過去の State 5 進入イベントを全抽出します ===")
    raw_events = []
    
    for idx, t in enumerate(tickers):
        price_path = AdvancedConfig.prices_dir / f"{t}.csv"
        if (idx + 1) % 500 == 0 or (idx + 1) == len(tickers):
            print(f"  [スキャン中] {idx+1} / {len(tickers)} 銘柄完了...")
            
        if not price_path.exists():
            continue
            
        try:
            df_raw = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()
            if len(df_raw) < 150:
                continue
                
            df_ind = analyzer.calculate_indicators(df_raw)
            evs = analyzer.simulate_states_and_collect_events(df_ind, t)
            
            # 各イベントについて未来データの計算を実行
            for ev in evs:
                entry_idx = ev["idx_pos"]
                entry_close = ev["entry_close"]
                
                # ① Forward Returns (先行きリターン)
                for period in AdvancedConfig.forward_periods:
                    target_idx = entry_idx + period
                    if target_idx < len(df_raw):
                        future_close = df_raw.iloc[target_idx]["Close"]
                        ev[f"return_{period}d"] = (future_close - entry_close) / entry_close * 100
                    else:
                        ev[f"return_{period}d"] = np.nan
                        
                # ② Maximum Drawdown until peak & 保有期間
                future_window = df_raw.iloc[entry_idx : entry_idx + 120]
                if len(future_window) > 1:
                    max_high_idx = future_window["High"].idxmax()
                    peak_idx = df_raw.index.get_loc(max_high_idx)
                    
                    ev["holding_days_to_peak"] = peak_idx - entry_idx
                    
                    peak_period_df = df_raw.iloc[entry_idx : peak_idx + 1]
                    if len(peak_period_df) > 1:
                        cum_max = peak_period_df["High"].cummax()
                        drawdowns = (peak_period_df["Low"] - cum_max) / cum_max * 100
                        ev["max_drawdown"] = drawdowns.min()
                    else:
                        ev["max_drawdown"] = 0.0
                else:
                    ev["holding_days_to_peak"] = np.nan
                    ev["max_drawdown"] = np.nan
                    
                raw_events.append(ev)
        except Exception:
            continue
            
    print(f"\n過去5年間の全データから 【 {len(raw_events)} 件 】 の State 5 進入イベントが検出されました。")
    if not raw_events:
        print("分析対象イベントが0件です。処理を終了します。")
        return
        
    df_evs = pd.DataFrame(raw_events)
    
    # ==========================================
    # 1. Forward Return Analysis
    # ==========================================
    print("\n--- 1. Forward Return Analysis の統計を集計中... ---")
    ret_cols = [f"return_{p}d" for p in AdvancedConfig.forward_periods]
    ret_summary = df_evs[ret_cols].describe(percentiles=[0.25, 0.5, 0.75]).transpose()
    ret_summary.to_csv(AdvancedConfig.output_dir / "state5_forward_returns.csv", encoding="utf-8-sig")
    
    # ==========================================
    # 2 & 3. Maximum Drawdown & Profit Factor / Expectancy Analysis
    # ==========================================
    print("--- 2 & 3. Drawdown & PF Analysis の統計を集計中... ---")
    df_eval = df_evs.dropna(subset=["return_60d"])
    
    win_events = df_eval[df_eval["return_60d"] > 0]
    loss_events = df_eval[df_eval["return_60d"] <= 0]
    
    win_rate = len(win_events) / len(df_eval) if len(df_eval) > 0 else 0.0
    avg_win = win_events["return_60d"].mean() if not win_events.empty else 0.0
    avg_loss = abs(loss_events["return_60d"].mean()) if not loss_events.empty else 0.0
    
    total_profit = win_events["return_60d"].sum() if not win_events.empty else 0.0
    total_loss = abs(loss_events["return_60d"].sum()) if not loss_events.empty else 1.0
    profit_factor = total_profit / total_loss if total_loss > 0 else 0.0
    
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    avg_dd = df_evs["max_drawdown"].mean()
    median_dd = df_evs["max_drawdown"].median()
    
    # 4. Holding Period (保有期間分析)
    avg_hold = df_evs["holding_days_to_peak"].mean()
    median_hold = df_evs["holding_days_to_peak"].median()

    # ==========================================
    # 5. State5 Sub Classification (K-Means クラスタリング)
    # ==========================================
    print("--- 5. K-Means による State 5 のサブクラス分類を実行中... ---")
    cluster_features = ["vol_ratio", "rsi14", "atr_ratio", "bb_width", "ma25_slope"]
    df_clust = df_evs.dropna(subset=cluster_features + ["return_60d"]).copy()
    
    if len(df_clust) >= 10:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_clust[cluster_features])
        
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df_clust["cluster"] = kmeans.fit_predict(X_scaled)
        
        cluster_summary_rows = []
        for c in range(3):
            c_data = df_clust[df_clust["cluster"] == c]
            c_win_rate = (c_data["return_60d"] > 0).mean() * 100
            c_ret_mean = c_data["return_60d"].mean()
            c_dd_mean = c_data["max_drawdown"].mean()
            
            c_vol = c_data["vol_ratio"].median()
            c_rsi = c_data["rsi14"].median()
            c_atr = c_data["atr_ratio"].median()
            c_bb = c_data["bb_width"].median()
            c_slope = c_data["ma25_slope"].median()
            
            cluster_summary_rows.append({
                "Cluster_ID": f"Type {c}",
                "Sample_Count": len(c_data),
                "Win_Rate_60d_pct": round(c_win_rate, 2),
                "Avg_Return_60d_pct": round(c_ret_mean, 2),
                "Avg_Max_Drawdown_pct": round(c_dd_mean, 2),
                "Median_Volume_Ratio": round(c_vol, 2),
                "Median_RSI": round(c_rsi, 1),
                "Median_ATR_Ratio": round(c_atr, 2),
                "Median_BB_Width": round(c_bb, 2),
                "Median_MA25_Slope": round(c_slope, 2)
            })
        df_cluster_sum = pd.DataFrame(cluster_summary_rows)
        df_cluster_sum.to_csv(AdvancedConfig.output_dir / "state5_sub_classification.csv", index=False, encoding="utf-8-sig")
    else:
        df_cluster_sum = pd.DataFrame()
        print("  [警告] クラスタリングに必要なサンプル数が足りません。")

    # ==========================================
    # レポートファイルの作成
    # ==========================================
    report_path = AdvancedConfig.output_dir / "state5_comprehensive_research.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("★ State 5 (ふるい落とし) 期待値・投資効率徹底検証レポート ★\n")
        f.write("==================================================\n")
        f.write(f"解析成功イベント総数: {len(df_evs)} 件\n\n")
        
        f.write("[1. Forward Return Analysis (先行き累積リターンの中央値推移)]\n")
        f.write("--------------------------------------------------\n")
        for p in AdvancedConfig.forward_periods:
            med_val = df_evs[f"return_{p}d"].median()
            mean_val = df_evs[f"return_{p}d"].mean()
            f.write(f"  ・進入後 {p:3d} 営業日後 ➔ 中央値: {med_val:+6.2f} % (平均: {mean_val:+6.2f} %)\n")
        f.write("--------------------------------------------------\n\n")
        
        f.write("[2. Risk & Profit Factor Analysis (リスク・投資期待値構造)]\n")
        f.write("※以下は 60営業日後（約3ヶ月）を基準とした損益効率です:\n")
        f.write("--------------------------------------------------\n")
        f.write(f"  ● 勝率 (60日後リターン > 0)    : {win_rate*100:.2f} %\n")
        f.write(f"  ● 平均利益率 (Win Events)     : {avg_win:+.2f} %\n")
        f.write(f"  ● 平均損失率 (Loss Events)    : -{avg_loss:.2f} %\n")
        f.write(f"  ● Profit Factor (プロフィット)  : {profit_factor:.2f}\n")
        f.write(f"  ● 統計的期待値 (Expectancy)     : {expectancy:+.2f} % (1取引あたりの期待収益率)\n")
        f.write(f"  ● 途中の平均最大下落率 (Max DD) : {avg_dd:.2f} % (中央値: {median_dd:.2f} %)\n")
        f.write("--------------------------------------------------\n\n")
        
        f.write("[3. Holding Period Analysis (高値ピークまでの保有期間)]\n")
        f.write("--------------------------------------------------\n")
        f.write(f"  ● 高値ピークをつけるまでの平均期間: {avg_hold:.1f} 営業日 (中央値: {median_hold:.1f} 営業日)\n")
        if not df_evs["holding_days_to_peak"].dropna().empty:
            f.write("  ● 保有期間の分布（件数割合）:\n")
            counts, bins = np.histogram(df_evs["holding_days_to_peak"].dropna(), bins=[0, 10, 20, 40, 60, 90, 120])
            for i in range(len(counts)):
                pct = counts[i] / len(df_evs["holding_days_to_peak"].dropna()) * 100
                bar = "■" * int(pct / 4)
                f.write(f"     - Day {bins[i]:3d} 〜 {bins[i+1]:3d} : {pct:5.1f} % {bar}\n")
        f.write("--------------------------------------------------\n\n")
        
        if not df_cluster_sum.empty:
            f.write("[4. State 5 Sub-Classification (K-Meansによる『最適仕込みType』の決定)]\n")
            f.write("※State 5に入った瞬間の特徴量の違いによって3つのクラスに分類:\n")
            f.write("--------------------------------------------------\n")
            for idx, row in df_cluster_sum.iterrows():
                f.write(f"  ■ 【{row['Cluster_ID']}】(サンプル数: {row['Sample_Count']} 件)\n")
                f.write(f"     ・勝率: {row['Win_Rate_60d_pct']:.2f} % / 60日後の平均期待リターン: {row['Avg_Return_60d_pct']:.2f} %\n")
                f.write(f"     ・特徴: 出来高倍率={row['Median_Volume_Ratio']}倍, RSI={row['Median_RSI']}%, BB幅={row['Median_BB_Width']}%\n")
                f.write("     ---------------------------------------------\n")
            f.write("\n  ● 【結論】：最も大化けする確率（勝率）が高い『黄金仕込みType』は、\n")
            best_type = df_cluster_sum.loc[df_cluster_sum["Win_Rate_60d_pct"].idxmax()]
            f.write(f"     ➔ 『{best_type['Cluster_ID']}』 です。\n")
            f.write(f"        (出来高が {best_type['Median_Volume_Ratio']}倍 付近で売り枯れており、RSIが {best_type['Median_RSI']}% の中立圏、BB幅が {best_type['Median_BB_Width']}% の状態で拾うタイプ)\n")
        f.write("==================================================\n")

    print(f"\n[解析成功]: 詳細多角検証レポートを保存しました: {report_path.name}")
    
    # 画面にレポートを表示
    with open(report_path, "r", encoding="utf-8") as f:
        print("\n" + f.read())


if __name__ == "__main__":
    main()