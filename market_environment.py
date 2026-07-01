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
        yfinanceのマルチインデックス(2行ヘッダー)を完全に解体し、
        エラー（No numeric types to aggregate）を100%根本から消滅させます。
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
                    df_test.sort_index()
                    
                    if (
                        len(df_test) >= 200 
                        and "Close" in df_test.columns 
                        and not df_test["Close"].isna().all() 
                        and not df_test.index.isnull().any()
                    ):
                        is_new = False
                except Exception:
                    is_new = True
            
            period = "5y" if is_new else "5d"
            if is_new:
                print(f"  [自己修復起動] インデックス {t} の破損・空データを検知したため、5年分を自動再構築します...")
            
            try:
                data = yf.download(t, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
                if data.empty:
                    continue
                
                # 【修正点1】：yfinanceのマルチインデックス（2行ヘッダー）を完全に解体・平坦化
                if isinstance(data.columns, pd.MultiIndex):
                    t_data = data[t].dropna()
                else:
                    t_data = data.dropna()
                    
                t_data = t_data[["Open", "High", "Low", "Close", "Volume"]]
                t_data.index = pd.to_datetime(t_data.index, errors="coerce")
                
                if not is_new and price_path.exists():
                    df_existing = pd.read_csv(price_path, index_col=0)
                    
                    # 既存のマルチインデックス破損データがあった場合の安全処理
                    if isinstance(df_existing.columns, pd.MultiIndex):
                        try:
                            df_existing = df_existing[t]
                        except KeyError:
                            pass
                    
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
            
            # 【修正点2】：既存のマルチインデックス破損データへの防衛策
            if isinstance(d.columns, pd.MultiIndex):
                d = d[main_index_ticker]
                
            d.index = pd.to_datetime(d.index, errors="coerce")
            d = d.dropna(how="all").sort_index()
            
            # 【修正点3】：全列を強制的に「数値型(float)」にキャスト（不純物を排除）
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in d.columns:
                    d[col] = pd.to_numeric(d[col], errors="coerce")
            
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
            print(f"  [デバッグ警告] 地合い判定中に不測のエラーが発生しました: {e}")
            return default_env
