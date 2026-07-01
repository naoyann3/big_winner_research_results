import pandas as pd
import yfinance as yf
from pathlib import Path
import numpy as np

class MarketEnvironmentManager:
    PRICES_DIR = Path("data_cache/prices")
    
    @classmethod
    def update_market_indices(cls):
        """
        TOPIX(^TPX)と日経平均(^N225)のデータをダウンロード・更新
        """
        cls.PRICES_DIR.mkdir(parents=True, exist_ok=True)
        tickers = ["^TPX", "^N225"]
        try:
            # 最新5日分をダウンロードしてマージ
            data = yf.download(tickers, period="5d", interval="1d", group_by="ticker", auto_adjust=True, progress=False, threads=False)
            for t in tickers:
                price_path = cls.PRICES_DIR / f"{t}.csv"
                if isinstance(data.columns, pd.MultiIndex):
                    t_data = data[t].dropna()
                else:
                    t_data = data.dropna()
                    
                if t_data.empty:
                    continue
                    
                t_data = t_data[["Open", "High", "Low", "Close", "Volume"]]
                if price_path.exists():
                    df_existing = pd.read_csv(price_path, index_col=0, parse_dates=True)
                    df_combined = pd.concat([df_existing, t_data])
                    df_combined = df_combined[~df_combined.index.duplicated(keep="last")].sort_index()
                else:
                    df_combined = t_data.sort_index()
                df_combined.to_csv(price_path, index=True, encoding="utf-8-sig")
        except Exception as e:
            print(f"  [警告] インデックスの更新中にエラーが発生しました: {e}")

    @classmethod
    def get_current_environment(cls, date_str: str) -> dict:
        """
        指定日の市場環境(TOPIX基準のBull/Bearなど)を判定して返します
        """
        cls.update_market_indices() # 常に最新の指数をダウンロード
        topix_path = cls.PRICES_DIR / "^TPX.csv"
        
        default_env = {
            "topix_close": 0.0,
            "market_state_topix": "Neutral"
        }
        
        if not topix_path.exists():
            return default_env
            
        try:
            d = pd.read_csv(topix_path, index_col=0, parse_dates=True).sort_index()
            if len(d) < 200:
                return default_env
                
            d["ma25"] = d["Close"].rolling(25).mean()
            d["ma200"] = d["Close"].rolling(200).mean()
            
            # 指定日、または最新日の行を取得
            if date_str in d.index.strftime("%Y-%m-%d"):
                row = d.loc[date_str]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
            else:
                row = d.iloc[-1]
                
            close = float(row["Close"])
            ma25 = float(row["ma25"])
            ma200 = float(row["ma200"])
            
            # トレンド状態の数理判定
            if close > ma25 > ma200:
                state = "Bull"
            elif close < ma25 < ma200:
                state = "Bear"
            elif ma25 < close < ma200 or ma200 < close < ma25:
                state = "Range"
            else:
                state = "Neutral"
                
            return {
                "topix_close": close,
                "market_state_topix": state
            }
        except Exception:
            return default_env