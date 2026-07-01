import pandas as pd
import yfinance as yf
from pathlib import Path
import numpy as np
import yaml

class MarketEnvironmentManager:
    PRICES_DIR = Path("data_cache/prices")
    CONFIG_FILE = Path("../config.yaml")  # 1つ上の親フォルダの config.yaml を指す

    @classmethod
    def load_config_tickers(cls) -> list[str]:
        """config.yamlから分析用ティッカーリストを動的に読み込みます"""
        # Playground直下、または同フォルダにある config.yaml を安全に探索
        config_path = cls.CONFIG_FILE if cls.CONFIG_FILE.exists() else Path("config.yaml")
        
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                    return cfg.get("research", {}).get("market_tickers", ["1306.T", "^N225"])
            except Exception:
                pass
        return ["1306.T", "^N225"]

    @classmethod
    def update_market_indices(cls, tickers: list[str]):
        """
        設定ファイルから読み込まれた市場インデックスをダウンロード・更新
        """
        cls.PRICES_DIR.mkdir(parents=True, exist_ok=True)
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
        指定日の市場環境(主たるインデックス基準)を判定して返します
        """
        # config.yaml から動的にリストを取得
        tickers = cls.load_config_tickers()
        cls.update_market_indices(tickers)
        
        # リストの最初のティッカー（今回は 1306.T）を環境判定用インデックスとする
        main_index_ticker = tickers[0] if tickers else "1306.T"
        main_index_path = cls.PRICES_DIR / f"{main_index_ticker}.csv"
        
        default_env = {
            "topix_close": 0.0,
            "market_state_topix": "Neutral"
        }
        
        if not main_index_path.exists():
            return default_env
            
        try:
            d = pd.read_csv(main_index_path, index_col=0, parse_dates=True).sort_index()
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
            
            # 判定ロジック
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
