import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# グラフ描画ライブラリのインポート
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# --- 設定 ---
class DynamicsConfig:
    prices_dir = Path("../data_cache/prices")
    fund_dir = Path("../data_cache/fundamentals")
    input_events_csv = Path("research_results/detected_big_winners.csv")
    output_dir = Path("research_results")
    
    # 毎日漏れなく切り出す（Day -30 から Day 0 までの全31営業日）
    relative_days = list(range(-30, 1))


class DynamicsFeatureExtractor:
    """
    Day -30 〜 Day 0 までの変化率・差分を算出し、時間窓を毎日抽出するクラス
    """
    def __init__(self, prices_dir: Path, fund_dir: Path):
        self.prices_dir = prices_dir
        self.fund_dir = fund_dir

    def calculate_indicators_with_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        絶対値に加えて、前日比（Growth Rate）や前日差（Delta）を一括計算
        """
        d = df.copy()
        
        # 移動平均と乖離・傾き
        for ma in [25, 75, 200]:
            d[f"ma{ma}"] = d["Close"].rolling(ma).mean()
            d[f"ma{ma}_dev"] = (d["Close"] - d[f"ma{ma}"]) / d[f"ma{ma}"] * 100
            
        # 傾き
        d["ma25_slope"] = d["ma25"].pct_change(5) * 100
        d["ma75_slope"] = d["ma75"].pct_change(5) * 100

        # 出来高倍率（20日平均比）
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

        # モメンタム & 52週高安
        delta = d["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        d["rsi14"] = 100 - (100 / (1 + (gain / np.where(loss > 0, loss, 1.0))))
        
        d["high_52w"] = d["High"].rolling(250, min_periods=50).max()
        d["low_52w"] = d["Low"].rolling(250, min_periods=50).min()
        d["dist_to_52w_high"] = (d["Close"] - d["high_52w"]) / d["high_52w"] * 100
        d["dist_to_52w_low"] = (d["Close"] - d["low_52w"]) / d["low_52w"] * 100

        # --------------------------------------------------
        # ★【新規追加】：前日差（Delta）と変化率（Growth Rate）★
        # --------------------------------------------------
        d["delta_volume_ratio"] = d["vol_ratio_20"].diff()
        d["volume_growth_rate"] = d["Volume"] / d["Volume"].shift(1)
        d["delta_ma25_slope"] = d["ma25_slope"].diff()
        d["delta_bb_width"] = d["bb_width"].diff()  # マイナスであるほどスクイーズが急加速
        d["delta_atr_ratio"] = d["atr_ratio"].diff()  # プラスであるほどボラが急拡大

        return d

    def extract_daily_features(self, ticker: str, day_0_date: datetime, event_meta: dict) -> list[dict]:
        price_path = self.prices_dir / f"{ticker}.csv"
        if not price_path.exists():
            return []
            
        try:
            df_raw = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()
            df = self.calculate_indicators_with_delta(df_raw)
            
            if day_0_date not in df.index:
                idx_pos = df.index.get_indexer([day_0_date], method="nearest")[0]
            else:
                idx_pos = df.index.get_loc(day_0_date)

            extracted_rows = []
            
            # Day -30 〜 Day 0 まで「毎日」漏れなく抽出
            for rel_day in DynamicsConfig.relative_days:
                target_idx = idx_pos + rel_day
                if target_idx < 0 or target_idx >= len(df):
                    continue
                    
                target_date = df.index[target_idx]
                row_data = df.iloc[target_idx]
                
                extracted_rows.append({
                    "ticker": ticker,
                    "event_day_0": day_0_date.strftime("%Y-%m-%d"),
                    "relative_day": rel_day,
                    "date": target_date.strftime("%Y-%m-%d"),
                    "multiplier": event_meta["multiplier"],
                    
                    # 毎日の中央値用（絶対値）
                    "vol_ratio_20": float(row_data["vol_ratio_20"]) if not pd.isna(row_data["vol_ratio_20"]) else None,
                    "ma25_dev": float(row_data["ma25_dev"]) if not pd.isna(row_data["ma25_dev"]) else None,
                    "ma75_dev": float(row_data["ma75_dev"]) if not pd.isna(row_data["ma75_dev"]) else None,
                    "ma200_dev": float(row_data["ma200_dev"]) if not pd.isna(row_data["ma200_dev"]) else None,
                    "ma25_slope": float(row_data["ma25_slope"]) if not pd.isna(row_data["ma25_slope"]) else None,
                    "ma75_slope": float(row_data["ma75_slope"]) if not pd.isna(row_data["ma75_slope"]) else None,
                    "bb_width": float(row_data["bb_width"]) if not pd.isna(row_data["bb_width"]) else None,
                    "atr_ratio": float(row_data["atr_ratio"]) if not pd.isna(row_data["atr_ratio"]) else None,
                    "rsi14": float(row_data["rsi14"]) if not pd.isna(row_data["rsi14"]) else None,
                    "dist_to_52w_high": float(row_data["dist_to_52w_high"]) if not pd.isna(row_data["dist_to_52w_high"]) else None,
                    "dist_to_52w_low": float(row_data["dist_to_52w_low"]) if not pd.isna(row_data["dist_to_52w_low"]) else None,
                    
                    # 新設された「変化率・変化速度」
                    "delta_volume_ratio": float(row_data["delta_volume_ratio"]) if not pd.isna(row_data["delta_volume_ratio"]) else None,
                    "volume_growth_rate": float(row_data["volume_growth_rate"]) if not pd.isna(row_data["volume_growth_rate"]) else None,
                    "delta_ma25_slope": float(row_data["delta_ma25_slope"]) if not pd.isna(row_data["delta_ma25_slope"]) else None,
                    "delta_bb_width": float(row_data["delta_bb_width"]) if not pd.isna(row_data["delta_bb_width"]) else None,
                    "delta_atr_ratio": float(row_data["delta_atr_ratio"]) if not pd.isna(row_data["delta_atr_ratio"]) else None,
                })
                
            return extracted_rows
        except Exception as e:
            print(f"Error extracting daily features for {ticker}: {e}")
            return []


def generate_plots(profile_df: pd.DataFrame):
    """
    可視化：Day-30 〜 Day0 までの主要指標の中央値推移をプロットして保存
    """
    if not HAS_MATPLOTLIB:
        print("\n[警告]: matplotlib が未インストールのため、グラフ画像の生成はスキップします。")
        return
        
    print("\n--- 3. 分析グラフの画像出力を開始します... ---")
    plt.rcParams["font.family"] = "sans-serif"
    
    # 4つの指標をプロット
    plots_info = [
        ("vol_ratio_20", "Median Volume Ratio (20d avg ratio)", "volume_ratio.png", "blue"),
        ("ma25_slope", "Median MA25 Slope (%)", "ma25_slope.png", "green"),
        ("bb_width", "Median Bollinger Band Width (%)", "bb_width.png", "red"),
        ("atr_ratio", "Median ATR Ratio (%)", "atr_ratio.png", "purple")
    ]
    
    for col, title, filename, color in plots_info:
        plt.figure(figsize=(10, 5))
        plt.plot(profile_df.index, profile_df[col], marker='o', color=color, linewidth=2, label=col)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Relative Days (Day -30 to Day 0)", fontsize=11)
        plt.ylabel("Median Value", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.axvline(x=0, color="gray", linestyle=":", label="Day 0 (Ignition)")
        plt.legend()
        plt.tight_layout()
        
        save_path = DynamicsConfig.output_dir / filename
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  [グラフ保存完了]: {save_path.name}")


def analyze_ignition_sequence(df: pd.DataFrame):
    """
    各大化けイベントにおいて、「最も急激に変化した日（前日差の絶対値が最大、または変化率最大）」
    の平均・中央値を算出し、市場点火の物理法則を解き明かします。
    """
    print("\n--- 4. 『市場の点火順序（Ignition Sequence）』の数理分析を開始します ---")
    
    # 各イベント（同じ ticker と event_day_0 の組み合わせ）ごとにグループ化
    grouped = df.groupby(["ticker", "event_day_0"])
    
    ignition_results = []
    
    # 追跡したい「変化」と、その評価基準（最大変化を探すカラム）
    change_targets = {
        "BB Width 収縮の極限（最もスクイーズが進んだ日）": ("bb_width", "min"),
        "出来高の急増（前日差が最大）": ("delta_volume_ratio", "max"),
        "出来高の前日比成長率（倍率が最大）": ("volume_growth_rate", "max"),
        "MA25傾きの急加速（前日差が最大）": ("delta_ma25_slope", "max"),
        "ボラティリティ（ATR）の急増（前日差が最大）": ("delta_atr_ratio", "max"),
        "RSIの最高加速（前日差が最大）": ("rsi14", "max_diff") # 簡易的にRSIは後で計算
    }
    
    events_count = 0
    for name, grp in grouped:
        if len(grp) < 15: # データ数が少なすぎるイベントはスキップ
            continue
        events_count += 1
        
        event_ignitions = {}
        
        # 各指標について、最大変化が発生した relative_day を特定する
        for label, (col, method) in change_targets.items():
            try:
                if col == "rsi14" and method == "max_diff":
                    # RSIの前日差が最大になる日
                    diff_rsi = grp["rsi14"].diff().abs()
                    max_idx = diff_rsi.idxmax()
                elif method == "max":
                    max_idx = grp[col].abs().idxmax()
                elif method == "min":
                    max_idx = grp[col].idxmin()
                    
                if pd.isna(max_idx):
                    continue
                
                # インデックスから relative_day の値を取得
                ign_day = grp.loc[max_idx, "relative_day"]
                event_ignitions[label] = ign_day
            except Exception:
                continue
                
        if event_ignitions:
            ignition_results.append(event_ignitions)
            
    # 全イベントの結果を集計
    df_ign = pd.DataFrame(ignition_results)
    mean_days = df_ign.mean().sort_values() # 平均して何日前に発生したか
    median_days = df_ign.median().sort_index()
    
    # 結果レポートの書き出し
    report_path = DynamicsConfig.output_dir / "ignition_sequence_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("★ AIが炙り出した『日本株大化けの物理法則・点火の順序』レポート ★\n")
        f.write("==================================================\n")
        f.write(f"精査対象大化けイベント総数: {events_count} 件\n")
        f.write("分析対象時系列: Day -30 〜 Day 0 (初動日) までの全31日間\n\n")
        
        f.write("[平均して初動の何営業日前に、どの変化が最大化したか（点火の順番順）]\n")
        f.write("--------------------------------------------------\n")
        for rank, (label, val) in enumerate(mean_days.items(), 1):
            f.write(f" 第 {rank} 段階: 【{label:40s}】 ➔ 平均して 【 Day {val:.2f} 】 (約 {abs(val):.1f} 営業日前)\n")
        f.write("--------------------------------------------------\n\n")
        
        f.write("■ クオンツ的解釈のヒント:\n")
        f.write("・「第1段階」に現れるイベントは、最も早くから水面下で起こり始めている『予兆』です。\n")
        f.write("・「第2〜3段階」は、マグマが噴き出し始める『過熱・兆候』です。\n")
        f.write("・「最後の段階」は、Day 0付近で発生する、大相場開始の『最終決定トリガー（点火）』です。\n")
        
    print(f"\n[解析成功]: 点火順序の物理法則レポートを保存しました: {report_path.name}")
    
    # 画面にも出力
    with open(report_path, "r", encoding="utf-8") as f:
        print("\n" + f.read())


# --- メイン実行処理 ---
def main():
    if not DynamicsConfig.input_events_csv.exists():
        print(f"エラー: {DynamicsConfig.input_events_csv} が見つかりません。big_winner_analyzer.py を先に実行してください。")
        return
        
    df_events = pd.read_csv(DynamicsConfig.input_events_csv)
    print(f"大化けイベントリストを読み込みました。イベント数: {len(df_events)} 件")
    
    extractor = DynamicsFeatureExtractor(DynamicsConfig.prices_dir, DynamicsConfig.fund_dir)
    
    print("\n=== Step 1 & 2: Day-30〜Day0 までの日次ダイナミクスデータを抽出します ===")
    dataset = []
    for idx, ev in enumerate(df_events.itertuples()):
        if (idx + 1) % 100 == 0 or (idx + 1) == len(df_events):
            print(f"  [データ抽出中] {idx+1} / {len(df_events)} イベント完了...")
            
        day_0 = pd.to_datetime(ev.day_0_date)
        rows = extractor.extract_daily_features(ev.ticker, day_0, ev._asdict())
        dataset.extend(rows)
        
    # 保存
    df_dataset = pd.DataFrame(dataset)
    dataset_path = DynamicsConfig.output_dir / "big_winner_dynamics_dataset.csv"
    df_dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    print(f"  [保存成功] 詳細ダイナミクスデータセット: {dataset_path.name}")
    
    # 毎日の中央値プロファイル
    print("\n=== Step 3: 日次の平均値・中央値プロファイルを集計します ===")
    numeric_cols = df_dataset.select_dtypes(include=[np.number]).columns.tolist()
    profile_cols = [c for c in numeric_cols if c not in ["multiplier"]]
    
    profile_median = df_dataset.groupby("relative_day")[profile_cols].median().sort_index()
    profile_median.to_csv(DynamicsConfig.output_dir / "big_winner_dynamics_profile_median.csv", encoding="utf-8-sig")
    print(f"  [保存成功] 毎日の中央値推移プロファイル: big_winner_dynamics_profile_median.csv")
    
    # 可視化グラフの出力
    generate_plots(profile_median)
    
    # 点火順序の物理法則解析
    analyze_ignition_sequence(df_dataset)


if __name__ == "__main__":
    main()