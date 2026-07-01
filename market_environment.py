import pandas as pd
import yfinance as yf
from pathlib import Path
import numpy as np
import yaml

class MarketEnvironmentManager:
    PRICES_DIR = Path("data_cache/prices")
    CONFIG_FILE = Path("../config.yaml")  # 1つ上の親フォルダ of config.yaml

    @classmethod
    def load_config_tickers(cls) -> list[str]:
        """config.yamlから分析用ティッカーリストを動的に読み込みます"""
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
        設定された市場インデックスをダウンロード・更新します。
        【自己修復(Self-Healing)極限強化設計】：
        既存のCSVファイルの『中身（Close列などの数値データ）』が1ミリでも破損、
        またはすべてNaNになっている場合、問答無用でファイルを完全自動破壊し、
        5年分のまっさらな新品データを自動再構築して強制修復します。
        """
        cls.PRICES_DIR.mkdir(parents=True, exist_ok=True)
        
        for t in tickers:
            price_path = cls.PRICES_DIR / f"{t}.csv"
            
            is_new = True
            if price_path.exists():
                try:
                    df_test = pd.read_csv(price_path, index_col=0)
                    df_test.index = pd.to_datetime(df_test.index, errors="coerce")
                    df_test = df_test.dropna(how="all")
                    
                    # 正常に日付ソートができるかテスト
                    df_test.sort_index()
                    
                    # 【極限強化】：Close（株価）列が存在し、且つ値がすべてNaNになっておらず、
                    # 且つデータが200行以上ある場合のみ、正常ファイルとしてマージ
                    if (
                        len(df_test) >= 200 
                        and "Close" in df_test.columns 
                        and not df_test["Close"].isna().all() 
                        and not df_test.index.isnull().any()
                    ):
                        is_new = False  # 完全に正常なファイルと判定
                except Exception:
                    is_new = True
            
            # 新規なら5年分(5y)、既にデータがあるなら差分(5d)のみ取得
            period = "5y" if is_new else "5d"
            if is_new:
                print(f"  [自己修復起動] インデックス {t} の中身の破損・空データを検知したため、5年分を自動再構築します...")
            
            try:
                data = yf.download(t, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
                if data.empty:
                    continue
                    
                t_data = data[["Open", "High", "Low", "Close", "Volume"]]
                t_data.index = pd.to_datetime(t_data.index, errors="coerce")
                
                if not is_new and price_path.exists():
                    df_existing = pd.read_csv(price_path, index_col=0)
                    df_existing.index = pd.to_datetime(df_existing.index, errors="coerce")
                    df_existing = df_existing.dropna(how="all")
                    
                    df_combined = pd.concat([df_existing, t_data])
                    df_combined.index = pd.to_datetime(df_combined.index, errors="coerce")
                    df_combined = df_combined[~df_combined.index.duplicated(keep="last")].sort_index()
                else:
                    df_combined = t_data.sort_index()
                    
                df_combined.to_csv(price_path, index=True, encoding="utf-8-sig")
            except Exception as e:
                print(f"  [警告] インデックス {t} の更新中にエラーが発生しました: {e}")

    @classmethod
    def get_current_environment(cls, date_str: str) -> dict:
        """
        指定日の市場環境(主たるインデックス基準)を判定して返します
        """
        tickers = cls.load_config_tickers()
        cls.update_market_indices(tickers)
        
        main_index_ticker = tickers[0] if tickers else "1306.T"
        main_index_path = cls.PRICES_DIR / f"{main_index_ticker}.csv"
        
        default_env = {
            "topix_close": 0.0,
            "market_state_topix": "Neutral"
        }
        
        if not main_index_path.exists():
            return default_env
            
        try:
            d = pd.read_csv(main_index_path, index_col=0)
            d.index = pd.to_datetime(d.index, errors="coerce")
            d = d.dropna(how="all").sort_index()
            
            if len(d) < 200:
                return default_env
                
            d["ma25"] = d["Close"].rolling(25).mean()
            d["ma200"] = d["Close"].rolling(200).mean()
            
            formatted_index = d.index.strftime("%Y-%m-%d")
            
            if date_str in formatted_index:
                row = d.loc[date_str]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
            else:
                row = d.iloc[-1]
                
            close = float(row["Close"])
            ma25 = float(row["ma25"])
            ma200 = float(row["ma200"])
            
            # トレンド判定
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
        except Exception as e:
            # 万が一不測のエラーが起きた場合は、画面にエラー名を表示します
            print(f"  [デバッグ警告] 地合い判定中に不測のエラーが発生しました: {e}")
            return default_env
