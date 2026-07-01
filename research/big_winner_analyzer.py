import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# --- 設定 ---
class AnalyzerConfig:
    cache_dir = Path("../data_cache")
    prices_dir = Path("../data_cache/prices")
    fund_dir = Path("../data_cache/fundamentals")
    output_dir = Path("research_results")  # これはそのまま（このフォルダ内に結果が出ます）
    
    # 大化け判定基準（例：120営業日（約半年）以内に最低2倍（+100%））
    holding_window = 120
    target_multiplier = 2.0

    # 取得したい時間窓（Day 0 を基準とした相対営業日数）
    relative_days = [-30, -20, -10, -5, -1, 0, 5, 10]


class BigWinnerEventFinder:
    """
    Step 1: データベースから大化け銘柄とその「Day 0（起点）」を検出するクラス
    """
    def __init__(self, prices_dir: Path):
        self.prices_dir = prices_dir

    def find_events(self, ticker: str) -> list[dict]:
        price_path = self.prices_dir / f"{ticker}.csv"
        if not price_path.exists():
            return []
            
        try:
            df = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()
            if len(df) < 250:
                return []
                
            events = []
            close = df["Close"]
            high = df["High"]
            low = df["Low"]
            volume = df["Volume"]
            
            # 各日について、120日先までの最大上昇率を測定
            for idx in range(len(df) - AnalyzerConfig.holding_window):
                start_close = close.iloc[idx]
                future_window = df.iloc[idx : idx + AnalyzerConfig.holding_window]
                max_future_high = future_window["High"].max()
                
                # 倍率クリア判定
                multiplier = max_future_high / start_close
                if multiplier >= AnalyzerConfig.target_multiplier:
                    # この急騰トレンドにおける最安値（底）の日を特定
                    min_idx_in_window = future_window["Low"].idxmin()
                    bottom_price = future_window.loc[min_idx_in_window, "Low"]
                    
                    # 底打ち後、最初に「出来高が急増（過去20日平均の1.5倍）して上昇が始まった日」を Day 0 とする
                    # 簡易的に、底から20日以内で出来高が跳ね上がった日を探索
                    search_range = df.loc[min_idx_in_window:].head(20)
                    search_range["vol_avg20"] = search_range["Volume"].rolling(20, min_periods=1).mean()
                    
                    day_0_date = None
                    for date_idx, row in search_range.iterrows():
                        if row["Volume"] >= row["vol_avg20"] * 1.5 and row["Close"] > row["Open"]:
                            day_0_date = date_idx
                            break
                            
                    if day_0_date is None:
                        day_0_date = min_idx_in_window  # 見つからなければ最安値日を代用
                        
                    # 重複イベントの登録を防止（同じ銘柄で近い日付のイベントは1つにまとめる）
                    if events and (day_0_date - events[-1]["day_0_date"]).days < 100:
                        continue
                        
                    events.append({
                        "ticker": ticker,
                        "day_0_date": day_0_date,
                        "multiplier": round(multiplier, 2),
                        "peak_price": max_future_high,
                        "bottom_price": bottom_price
                    })
            return events
        except Exception as e:
            print(f"Error finding events for {ticker}: {e}")
            return []


class FeatureExtractor:
    """
    Step 2 & 3: 検出された Day 0 の前後（タイムウィンドウ）から、
    保守性・拡張性の高い「プラグイン方式」で特徴量を抽出するクラス
    """
    def __init__(self, prices_dir: Path, fund_dir: Path):
        self.prices_dir = prices_dir
        self.fund_dir = fund_dir

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        プラグインを適用する前の「基礎指標」を一括計算する（Pandas/Numpyで完結）
        """
        d = df.copy()
        
        # 移動平均
        for ma in [5, 10, 25, 50, 75, 100, 200]:
            d[f"ma{ma}"] = d["Close"].rolling(ma).mean()
            # 傾き（直近5日間の変化率）
            d[f"ma{ma}_slope"] = d[f"ma{ma}"].pct_change(5) * 100
            # 乖離率
            d[f"ma{ma}_dev"] = (d["Close"] - d[f"ma{ma}"]) / d[f"ma{ma}"] * 100

        # 出来高
        d["vol_avg20"] = d["Volume"].rolling(20).mean()
        d["vol_ratio_20"] = d["Volume"] / d["vol_avg20"]
        
        # ボラティリティ (ATR)
        high_low = d["High"] - d["Low"]
        high_cp = (d["High"] - d["Close"].shift(1)).abs()
        low_cp = (d["Low"] - d["Close"].shift(1)).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        d["atr"] = tr.rolling(14).mean()
        d["atr_ratio"] = d["atr"] / d["Close"] * 100
        
        # ボリンジャーバンド幅
        std20 = d["Close"].rolling(20).std()
        d["bb_width"] = (std20 * 4) / d["ma25"] * 100

        # RSI (14)
        delta = d["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / np.where(loss > 0, loss, 1.0)
        d["rsi14"] = 100 - (100 / (1 + rs))
        
        # 52週（250営業日）高値・安値からの比率
        d["high_52w"] = d["High"].rolling(250, min_periods=50).max()
        d["low_52w"] = d["Low"].rolling(250, min_periods=50).min()
        d["dist_to_52w_high"] = (d["Close"] - d["high_52w"]) / d["high_52w"] * 100
        d["dist_to_52w_low"] = (d["Close"] - d["low_52w"]) / d["low_52w"] * 100

        return d

    def extract_features_at_window(self, ticker: str, day_0_date: datetime, event_meta: dict) -> list[dict]:
        """
        Day 0 の前後（例: -30日〜+10日）の切り出しと特徴量マッピング
        """
        price_path = self.prices_dir / f"{ticker}.csv"
        fund_path = self.fund_dir / f"{ticker}.json"
        
        if not price_path.exists():
            return []
            
        try:
            df_raw = pd.read_csv(price_path, index_col=0, parse_dates=True).sort_index()
            df = self.calculate_technical_indicators(df_raw)
            
            # 時系列における Day 0 の位置（インデックス）を取得
            if day_0_date not in df.index:
                # 完全に一致する日付がない場合は一番近い営業日を採用
                idx_pos = df.index.get_indexer([day_0_date], method="nearest")[0]
            else:
                idx_pos = df.index.get_loc(day_0_date)

            # ファンダメンタルズデータのロード
            fund_data = {}
            if fund_path.exists():
                try:
                    with open(fund_path, "r", encoding="utf-8") as f:
                        fund_data = json.load(f)
                except Exception:
                    pass

            extracted_rows = []
            
            # 各時間窓（例：-30, -10, 0, +10営業日）を切り出し
            for rel_day in AnalyzerConfig.relative_days:
                target_idx = idx_pos + rel_day
                if target_idx < 0 or target_idx >= len(df):
                    continue
                    
                target_date = df.index[target_idx]
                row_data = df.iloc[target_idx]
                
                # 【保守性を重視した特徴量辞書の構築】
                # ここに辞書キーを追加するだけで、後からいくらでも特徴量を増やせます
                feature_set = {
                    "ticker": ticker,
                    "event_day_0": day_0_date.strftime("%Y-%m-%d"),
                    "relative_day": rel_day,  # -30, 0, 10 など
                    "date": target_date.strftime("%Y-%m-%d"),
                    "multiplier": event_meta["multiplier"],
                    
                    # --- 【価格特徴量】 ---
                    "close": float(row_data["Close"]),
                    "volume": float(row_data["Volume"]),
                    "dist_to_52w_high": float(row_data["dist_to_52w_high"]) if not pd.isna(row_data["dist_to_52w_high"]) else None,
                    "dist_to_52w_low": float(row_data["dist_to_52w_low"]) if not pd.isna(row_data["dist_to_52w_low"]) else None,
                    
                    # --- 【移動平均特徴量】 ---
                    "ma25_dev": float(row_data["ma25_dev"]) if not pd.isna(row_data["ma25_dev"]) else None,
                    "ma75_dev": float(row_data["ma75_dev"]) if not pd.isna(row_data["ma75_dev"]) else None,
                    "ma200_dev": float(row_data["ma200_dev"]) if not pd.isna(row_data["ma200_dev"]) else None,
                    "ma25_slope": float(row_data["ma25_slope"]) if not pd.isna(row_data["ma25_slope"]) else None,
                    
                    # --- 【出来高特徴量】 ---
                    "vol_ratio_20": float(row_data["vol_ratio_20"]) if not pd.isna(row_data["vol_ratio_20"]) else None,
                    
                    # --- 【ボラティリティ・モメンタム】 ---
                    "bb_width": float(row_data["bb_width"]) if not pd.isna(row_data["bb_width"]) else None,
                    "atr_ratio": float(row_data["atr_ratio"]) if not pd.isna(row_data["atr_ratio"]) else None,
                    "rsi14": float(row_data["rsi14"]) if not pd.isna(row_data["rsi14"]) else None,
                    
                    # --- 【ファンダメンタルズ（固定値）】 ---
                    "market_cap": fund_data.get("market_cap"),
                    "roe_pct": fund_data.get("roe_pct"),
                    "sector": fund_data.get("sector", "不明"),
                    "industry": fund_data.get("industry", "不明")
                }
                
                extracted_rows.append(feature_set)
                
            return extracted_rows
        except Exception as e:
            print(f"Error extracting features for {ticker}: {e}")
            return []


# --- メイン実行処理 ---
def main():
    AnalyzerConfig.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 全上場銘柄のリスト取得
    universe_path = Path("../universe.csv")  # ← ここに「../」を追加します
    if not universe_path.exists():
        print("universe.csv が見つかりません。")
        return
        
    df_uni = pd.read_csv(universe_path)
    tickers = df_uni["ticker"].dropna().tolist()
    
    # yfinanceの標準表記（.T）へ変換
    def quick_normalize(t):
        t = str(t).strip()
        return f"{t}.T" if "." not in t else t
    tickers = [quick_normalize(t) for t in tickers]

    # インスタンス化
    finder = BigWinnerEventFinder(AnalyzerConfig.prices_dir)
    extractor = FeatureExtractor(AnalyzerConfig.prices_dir, AnalyzerConfig.fund_dir)
    
    print("=== Step 1: 大化け銘柄と起点（Day 0）の探索を開始します ===")
    all_events = []
    for idx, t in enumerate(tickers):
        if (idx + 1) % 500 == 0 or (idx + 1) == len(tickers):
            print(f"  [探索中] {idx+1} / {len(tickers)} 銘柄精査完了...")
        
        events = finder.find_events(t)
        if events:
            all_events.extend(events)
            
    print(f"\n探索完了：過去5年間で大化けしたイベントが合計 【 {len(all_events)} 件 】 検出されました。")
    
    # 検出結果を一旦保存
    if not all_events:
        print("大化けイベントが1件も検出されませんでした。設定値（AnalyzerConfig.target_multiplier）を緩めてみてください。")
        return
        
    df_events = pd.DataFrame(all_events)
    df_events.to_csv(AnalyzerConfig.output_dir / "detected_big_winners.csv", index=False, encoding="utf-8-sig")

    print("\n=== Step 2 & 3: 特徴量抽出（時間窓マッピング）を開始します ===")
    dataset = []
    for idx, ev in enumerate(all_events):
        if (idx + 1) % 100 == 0 or (idx + 1) == len(all_events):
            print(f"  [マッピング中] {idx+1} / {len(all_events)} イベント完了...")
            
        day_0 = pd.to_datetime(ev["day_0_date"])
        rows = extractor.extract_features_at_window(ev["ticker"], day_0, ev)
        dataset.extend(rows)
        
    # 解析用総合データセットの保存
    df_dataset = pd.DataFrame(dataset)
    dataset_path = AnalyzerConfig.output_dir / "big_winner_features_dataset.csv"
    df_dataset.to_csv(dataset_path, index=False, encoding="utf-8-sig")
    
    print(f"\n【データセット構築成功】: {dataset_path} に保存しました。")
    print(f"データセット行数（イベント数 × 時間窓数）: {len(df_dataset)} 行")
    print("\n[次のフェーズ]: このCSVを利用して、Step 4の統計解析や機械学習（重要度算出）に進みます。")


if __name__ == "__main__":
    main()