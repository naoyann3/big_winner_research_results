import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
from pathlib import Path
import time
import random
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# --- 設定クラス ---
class ResearchConfig:
    years = 5
    min_daily_turnover_million = 10.0  # 最低1,000万円以上の売買代金
    cache_dir = Path("data_cache")

# --- 定数定義 ---
UNIVERSE_CSV = Path("universe.csv")
PRICES_DIR = ResearchConfig.cache_dir / "prices"
FUND_DIR = ResearchConfig.cache_dir / "fundamentals"

# GitHub Secretsから環境変数をロード
WEBHOOK_URL = os.environ.get("SPREADSHEET_WEBHOOK_URL")
GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_APP_PASSWORD")
NOTIFICATION_EMAIL = os.environ.get("NOTIFICATION_EMAIL")


def normalize_ticker(raw: str) -> str:
    ticker = str(raw).strip().upper()
    if not ticker:
        return ticker
    if "." not in ticker and not ticker.isdigit():
        ticker = f"{ticker}.T"
    return ticker


def run_with_timeout(func, timeout_sec: int, *args, **kwargs):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout_sec)
    except FuturesTimeoutError:
        future.cancel()
        return None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def fetch_fundamentals(ticker: str) -> dict | None:
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return None

    if not info:
        return None

    market_cap = info.get("marketCap")
    roe = info.get("returnOnEquity")
    profit_margin = info.get("profitMargins")
    revenue_growth = info.get("revenueGrowth")
    sector = info.get("sector")
    industry = info.get("industry")

    return {
        "market_cap": float(market_cap) if market_cap is not None else None,
        "roe_pct": float(roe) * 100 if roe is not None else None,
        "profit_margin_pct": float(profit_margin) * 100 if profit_margin is not None else None,
        "revenue_growth_pct": float(revenue_growth) * 100 if revenue_growth is not None else None,
        "sector": sector,
        "industry": industry,
    }


def _download_and_merge(tickers: list[str], period: str, is_new: bool):
    """
    株価データを一括ダウンロードし、マージするヘルパー関数
    """
    batch_size = 300
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"  バッチ取得中: {i+1}〜{min(i+batch_size, len(tickers))} / {len(tickers)}")
        
        try:
            # threads=FalseでSQLiteの競合エラーを100%回避します
            data = yf.download(batch, period=period, interval="1d", group_by="ticker", auto_adjust=True, progress=False, threads=False)
            
            for t in batch:
                price_path = PRICES_DIR / f"{t}.csv"
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        t_data = data[t].dropna()
                    else:
                        t_data = data.dropna()
                        
                    if t_data.empty:
                        continue
                        
                    t_data = t_data[["Open", "High", "Low", "Close", "Volume"]]
                    
                    # 新規の場合はそのまま保存、既存の場合はロードして最新日付をマージ
                    if is_new or not price_path.exists():
                        df_combined = t_data.sort_index()
                    else:
                        df_existing = pd.read_csv(price_path, index_col=0, parse_dates=True)
                        df_combined = pd.concat([df_existing, t_data])
                        df_combined = df_combined[~df_combined.index.duplicated(keep="last")].sort_index()
                        
                    df_combined.to_csv(price_path, index=True, encoding="utf-8-sig")
                except Exception:
                    continue
        except Exception as e:
            print(f"  バッチダウンロード中にエラーが発生しました: {e}")
            continue


def update_prices_daily(tickers: list[str]):
    """
    株価データの差分/新規構築を判断して実行します。
    """
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 既存のキャッシュ有無で銘柄を切り分け
    existing_tickers = []
    new_tickers = []
    
    for t in tickers:
        if (PRICES_DIR / f"{t}.csv").exists():
            existing_tickers.append(t)
        else:
            new_tickers.append(t)
            
    # ① キャッシュが存在しない銘柄（新規実行時など）：5年分（1945日）を落として自動構築
    if new_tickers:
        print(f"\n新規銘柄（キャッシュなし）の5年分株価ダウンロードを開始します... (対象: {len(new_tickers)} 銘柄)")
        period_days = max(365 * ResearchConfig.years + 120, 365)
        period_new = f"{period_days}d"
        _download_and_merge(new_tickers, period_new, is_new=True)
        
    # ② キャッシュがすでに存在する銘柄：最新5日分を落として末尾に差分マージ
    if existing_tickers:
        print(f"\n既存銘柄の差分更新（最新5日分）を開始します... (対象: {len(existing_tickers)} 銘柄)")
        _download_and_merge(existing_tickers, "5d", is_new=False)


def run_screener(tickers: list[str]) -> list[dict]:
    """
    全銘柄のキャッシュから、本日にスナイパー合格シグナルが出ているものをスクリーニングします。
    """
    print("\n--- 2. スクリーニングの計算を開始します ---")
    qualified_signals = []
    
    for idx, t in enumerate(tickers):
        price_path = PRICES_DIR / f"{t}.csv"
        fund_path = FUND_DIR / f"{t}.json"
        
        if (idx + 1) % 500 == 0 or (idx + 1) == len(tickers):
            print(f"  [計算中] {idx+1} / {len(tickers)} 銘柄...")
            
        if not price_path.exists():
            continue
            
        try:
            d = pd.read_csv(price_path, index_col=0, parse_dates=True)
            if len(d) < 220:
                continue
                
            # 基本指標計算
            d["ma25"] = d["Close"].rolling(25).mean()
            d["ma75"] = d["Close"].rolling(75).mean()
            d["ma200"] = d["Close"].rolling(200).mean()
            d["vol_avg20"] = d["Volume"].rolling(20).mean()
            d["volume_ratio_20"] = d["Volume"] / d["vol_avg20"]
            d["turnover_million"] = (d["Close"] * d["Volume"]) / 1_000_000
            d["turnover_avg20_million"] = d["turnover_million"].rolling(20).mean()
            d["ma_congestion_width_pct"] = (
                (d[["ma25", "ma75", "ma200"]].max(axis=1) - d[["ma25", "ma75", "ma200"]].min(axis=1))
                / d[["ma25", "ma75", "ma200"]].mean(axis=1) * 100
            )
            
            # スナイパー条件の計算
            d["ma_squeeze_20d"] = d["ma_congestion_width_pct"].rolling(20).max() <= 5.0
            
            vol_avg20_prior = d["Volume"].shift(1).rolling(20).mean()
            vol_avg5_prior = d["Volume"].shift(1).rolling(5).mean()
            d["dry_up"] = np.where(vol_avg20_prior > 0, vol_avg5_prior < (vol_avg20_prior * 0.8), False)
            
            d["recent_high_20d"] = d["High"].shift(1).rolling(20).max()
            d["breakout_20d"] = d["Close"] > d["recent_high_20d"]
            
            d["candle_mid_high"] = np.where(d["High"] > d["Low"], (d["Close"] - d["Low"]) / (d["High"] - d["Low"]) >= 0.5, False)
            d["is_positive_candle"] = d["Close"] > d["Open"]
            d["liquidity_ok"] = d["turnover_avg20_million"] >= ResearchConfig.min_daily_turnover_million
            
            # 追加フィルター：RSI(14) の算出
            delta = d["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / np.where(loss > 0, loss, 1.0)
            d["rsi14"] = 100 - (100 / (1 + rs))
            
            # スナイパーブレイクアウトシグナル
            d["sniper_breakout"] = (
                d["ma_squeeze_20d"] & d["dry_up"] & d["breakout_20d"] &
                (d["volume_ratio_20"] >= 1.5) & d["candle_mid_high"] & d["is_positive_candle"] & d["liquidity_ok"]
            )
            
            # 本日シグナルが立ち上がったばかり（エントリー初動） 且つ RSIが50以上、80未満（過熱すぎない）
            d["sniper_entry"] = d["sniper_breakout"] & ~d["sniper_breakout"].shift(1).fillna(False)
            d["qualified_signal"] = d["sniper_entry"] & (d["rsi14"] >= 50.0) & (d["rsi14"] < 80.0)
            
            # 最新の行（本日）で判定
            latest_row = d.iloc[-1]
            if bool(latest_row["qualified_signal"]):
                fundamentals = None
                if fund_path.exists():
                    try:
                        with open(fund_path, "r", encoding="utf-8") as f:
                            fundamentals = json.load(f)
                    except Exception:
                        pass
                
                qualified_signals.append({
                    "date": d.index[-1].strftime("%Y-%m-%d"),
                    "ticker": t,
                    "name": t,  # 銘柄名はあとで補完
                    "close": float(latest_row["Close"]),
                    "rsi": round(float(latest_row["rsi14"]), 1),
                    "sector": fundamentals.get("sector") if fundamentals else "不明",
                    "industry": fundamentals.get("industry") if fundamentals else "不明"
                })
        except Exception:
            continue
            
    return qualified_signals


def notify_results(signals: list[dict], universe: pd.DataFrame):
    """
    検出された合格シグナルを、GoogleスプレッドシートとGmailへ通知します。
    """
    if not signals:
        print("\n本日、合格基準を満たす初動スナイパーシグナルは発生しませんでした。")
        return
        
    # 銘柄名の自動マッピング
    name_map = dict(zip(universe["ticker"], universe["name"]))
    for sig in signals:
        sig["name"] = name_map.get(sig["ticker"], sig["ticker"])
        
    # 1. Googleスプレッドシートへの自動追記 (GAS Webhook)
    if WEBHOOK_URL:
        try:
            res = requests.post(WEBHOOK_URL, json={"signals": signals}, headers={"Content-Type": "application/json"})
            if res.status_code == 200:
                print("Googleスプレッドシートへの自動追記に成功しました。")
            else:
                print(f"スプレッドシート連携エラー: {res.text}")
        except Exception as e:
            print(f"スプレッドシート連携に失敗しました: {e}")
            
    # 2. メール通知 (Gmail SMTP経由)
    if GMAIL_USER and GMAIL_PASS and NOTIFICATION_EMAIL:
        try:
            msg = MIMEMultipart()
            msg["From"] = GMAIL_USER
            msg["To"] = NOTIFICATION_EMAIL
            msg["Subject"] = f"【スナイパー合格検知】大相場初動シグナル ({signals[0]['date']})"
            
            body = "本日、合格基準を満たした大相場初動の銘柄を検知しました。\n\n"
            body += "=========================================\n"
            for sig in signals:
                body += f"■ {sig['name']} ({sig['ticker']})\n"
                body += f"  ・終値: {sig['close']} 円\n"
                body += f"  ・RSI(14): {sig['rsi']} %\n"
                body += f"  ・セクター: {sig['sector']} / 業界: {sig['industry']}\n"
                body += "=========================================\n"
            body += "\n※スプレッドシートにもデータが自動追記されました。翌朝の寄り付き前の気配値に注目してください。\n"
            
            msg.attach(MIMEText(body, "plain", "utf-8"))
            
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(GMAIL_USER, GMAIL_PASS)
                server.sendmail(GMAIL_USER, [NOTIFICATION_EMAIL], msg.as_string())
            print("合格通知メールをご登録アドレス宛てに送信しました。")
        except Exception as e:
            print(f"メール送信に失敗しました: {e}")


def main():
    if not UNIVERSE_CSV.exists():
        print("universe.csv が見つかりません。")
        return
        
    df_uni = pd.read_csv(UNIVERSE_CSV)
    df_uni["ticker"] = df_uni["ticker"].map(normalize_ticker)
    tickers = df_uni["ticker"].dropna().tolist()
    
    # ① キャッシュデータバンクを本日時点の最新に更新・新規作成
    update_prices_daily(tickers)
    
    # ② スクリーニング
    signals = run_screener(tickers)
    
    # ③ 連携とメール送信
    notify_results(signals, df_uni)


if __name__ == "__main__":
    main()