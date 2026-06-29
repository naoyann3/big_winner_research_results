import os
from pathlib import Path
import pandas as pd
import numpy as np

# 機械学習ライブラリ (scikit-learn) のインポート
try:
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# --- 設定 ---
DATASET_CSV = Path("research_results/big_winner_features_dataset.csv")
OUTPUT_DIR = Path("research_results")


def analyze_time_series_profile(df: pd.DataFrame):
    """
    1. 時系列プロファイル分析
    相対営業日数（-30日前〜+10日後）ごとの特徴量の中央値・平均値を算出します
    """
    print("\n--- 1. 時系列プロファイル分析を実行中... ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 銘柄分析に不要な列を除外
    exclude_cols = ["multiplier", "close", "volume"]
    analysis_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    # relative_day（-30, -20, 0, 10等）ごとに各指標の中央値（Median）を集計
    profile_median = df.groupby("relative_day")[analysis_cols].median().sort_index()
    profile_mean = df.groupby("relative_day")[analysis_cols].mean().sort_index()
    
    # 保存
    profile_median.to_csv(OUTPUT_DIR / "big_winner_profile_median.csv", encoding="utf-8-sig")
    profile_mean.to_csv(OUTPUT_DIR / "big_winner_profile_mean.csv", encoding="utf-8-sig")
    print(f"  [成功] 時系列中央値レポートをエクスポートしました: big_winner_profile_median.csv")


def analyze_feature_importance_ml(df: pd.DataFrame):
    """
    2. 機械学習による「初動（Day 0）」と「平時（Day -30）」の識別と重要度算出
    """
    if not HAS_SKLEARN:
        print("\n[警告]: scikit-learn がインストールされていないため、機械学習による重要度分析はスキップします。")
        print("実行するには `pip install scikit-learn` を行ってください。")
        return

    print("\n--- 2. 機械学習（Random Forest）による初動決定因子分析を実行中... ---")
    
    # 「平時の状態（Day -30）」と「初動（Day 0）」のデータのみを抽出して比較する
    ml_data = df[df["relative_day"].isin([-30, 0])].copy()
    if ml_data.empty:
        print("  [エラー] 分析に必要な相対日データ（-30, 0）がデータセットに存在しません。")
        return
        
    # 目的変数: Day 0 の場合は 1 (初動)、Day -30 の場合は 0 (平時) と定義
    ml_data["is_day_0"] = np.where(ml_data["relative_day"] == 0, 1, 0)
    
    # 学習に使用する特徴量の選定（テキストデータやリークを招く株価そのものを除外）
    feature_cols = [
        "dist_to_52w_high", "dist_to_52w_low", 
        "ma25_dev", "ma75_dev", "ma200_dev", "ma25_slope",
        "vol_ratio_20", "bb_width", "atr_ratio", "rsi14"
    ]
    
    # 欠損値の補正（中央値埋め）
    X = ml_data[feature_cols].copy()
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
        
    y = ml_data["is_day_0"]

    # ランダムフォレストによる学習（初動と平時を分類させる）
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # 特徴量の重要度（Feature Importance）を抽出
    importances = rf.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance_Score": importances
    }).sort_values(by="Importance_Score", ascending=False)
    
    # 保存
    importance_df.to_csv(OUTPUT_DIR / "big_winner_feature_importance_ml.csv", index=False, encoding="utf-8-sig")
    
    print("\n==================================================")
    print("★ AIが暴いた「大化け株の初動（Day 0）」に最も寄与した特徴量ランキング ★")
    print("==================================================")
    for idx, row in importance_df.iterrows():
        print(f" {idx+1:2d}位: {row['Feature']:20s} (スコア: {row['Importance_Score']:.4f})")
    print("==================================================")
    print(f"  [成功] 機械学習レポートを保存しました: big_winner_feature_importance_ml.csv")


def main():
    if not DATASET_CSV.exists():
        print(f"エラー: {DATASET_CSV} が見つかりません。big_winner_analyzer.py を先に実行してください。")
        return
        
    # データセットのロード
    df = pd.read_csv(DATASET_CSV)
    print(f"データセットをロードしました。行数: {len(df)} 行")
    
    # ① 統計プロファイルの算出
    analyze_time_series_profile(df)
    
    # ② 機械学習による決定因子ランキング
    analyze_feature_importance_ml(df)


if __name__ == "__main__":
    main()