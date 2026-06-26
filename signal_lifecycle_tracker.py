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

# --- 設定 ---
class TrackerConfig:
    years = 5
    min_daily_turnover_million = 10.0  # 最低売買代金（1,000万円以上）
    tracking_days_limit = 60           # 追跡する最大営業日数（60日または90日）
    cache_dir = Path("data_cache")
    db_file = Path("signals_lifecycle.csv")          # 累積シグナル追跡台帳
    report_file = Path("signal_performance_report.csv") # 自動改善レポート
    output_dir = Path("big_winner_research_results")

# 環境変数のロード
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


def fetch_fundamentals(ticker: str) -> dict | None:
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        return None
    if not info:
        return None
    return {
        "market_cap": float(info.get("marketCap")) if info.get("marketCap") is not None else None,
        "roe_pct": float(info.get("returnOnEquity")) * 100 if info.get("returnOnEquity") is not None else None,
        "profit_margin_pct": float(info.get("profitMargins")) * 100 if info.get("profitMargins") is not None else None,
        "revenue_growth_pct": float(info.get("revenueGrowth")) * 100 if info.get("revenueGrowth") is not None else None,
        "sector": info.get("sector"),
        "industry": info.get("industry")
    }


def update_prices_daily(tickers: list[str]):
    """
    最新5日分を一括取得し、既存キャッシュに重複なくマージ（Threads=FalseでSQLite競合を回避）
    """
    prices_dir = TrackerConfig.cache_dir / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    
    period = "5d"
    batch_size = 300
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"  差分取得中: {i+1}〜{min(i+batch_size, len(tickers))} / {len(tickers)}")
        
        try:
            data = yf.download(batch, period=period, interval="1d", group_by="ticker", auto_adjust=True, progress=False, threads=False)
            
            for t in batch:
                price_path = prices_dir / f"{t}.csv"
                try:
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
                except Exception:
                    continue
        except Exception:
            continue


def _download_and_merge(tickers: list[str], period: str, prices_dir: Path, is_new: bool):
    """
    株価データを一括ダウンロードし、マージするヘルパー関数
    """
    batch_size = 300
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"  バッチ取得中: {i+1}〜{min(i+batch_size, len(tickers))} / {len(tickers)}")
        
        try:
            data = yf.download(batch, period=period, interval="1d", group_by="ticker", auto_adjust=True, progress=False, threads=False)
            
            for t in batch:
                price_path = prices_dir / f"{t}.csv"
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        t_data = data[t].dropna()
                    else:
                        t_data = data.dropna()
                        
                    if t_data.empty:
                        continue
                        
                    t_data = t_data[["Open", "High", "Low", "Close", "Volume"]]
                    
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
            print(f"  バッチダウンロードエラー: {e}")
            continue


def update_prices_daily_full(tickers: list[str]):
    """
    株価データの差分/新規構築を判断して実行します。
    """
    prices_dir = TrackerConfig.cache_dir / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    
    # 既存のキャッシュ有無で銘柄を切り分け
    existing_tickers = []
    new_tickers = []
    
    for t in tickers:
        if (prices_dir / f"{t}.csv").exists():
            existing_tickers.append(t)
        else:
            new_tickers.append(t)
            
    # ① キャッシュが存在しない銘柄（新規実行時など）：5年分を落として自動構築
    if new_tickers:
        print(f"\n新規銘柄（キャッシュなし）の5年分株価ダウンロードを開始します... (対象: {len(new_tickers)} 銘柄)")
        period_days = max(365 * TrackerConfig.years + 120, 365)
        period_new = f"{period_days}d"
        _download_and_merge(new_tickers, period_new, prices_dir, is_new=True)
        
    # ② キャッシュがすでに存在する銘柄：最新5日分を落として末尾に差分マージ
    if existing_tickers:
        print(f"\n既存銘柄の差分更新（最新5日分）を開始します... (対象: {len(existing_tickers)} 銘柄)")
        _download_and_merge(existing_tickers, "5d", prices_dir, is_new=False)


def run_screener_for_today(tickers: list[str], market_median_close: pd.Series) -> list[dict]:
    """
    本日に「Sniper Motion」シグナルが発生した合格銘柄を検出します。
    """
    signals = []
    prices_dir = TrackerConfig.cache_dir / "prices"
    fund_dir = TrackerConfig.cache_dir / "fundamentals"
    
    for t in tickers:
        price_path = prices_dir / f"{t}.csv"
        fund_path = fund_dir / f"{t}.json"
        
        if not price_path.exists():
            continue
            
        try:
            d = pd.read_csv(price_path, index_col=0, parse_dates=True)
            if len(d) < 220:
                continue
                
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
            
            d["ma_squeeze_20d"] = d["ma_congestion_width_pct"].rolling(20).max() <= 5.0
            
            vol_avg20_prior = d["Volume"].shift(1).rolling(20).mean()
            vol_avg5_prior = d["Volume"].shift(1).rolling(5).mean()
            d["dry_up"] = np.where(vol_avg20_prior > 0, vol_avg5_prior < (vol_avg20_prior * 0.8), False)
            
            d["recent_high_20d"] = d["High"].shift(1).rolling(20).max()
            d["is_breakout"] = d["Close"] > d["recent_high_20d"]
            d["breakout_age"] = d["is_breakout"].groupby((~d["is_breakout"]).cumsum()).cumsum()
            
            d["vol_velocity"] = d["Volume"].rolling(5).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 5 else 0.0, raw=True)
            d["vol_acceleration"] = d["vol_velocity"].diff()
            
            d["ma25_velocity"] = (d["ma25"] - d["ma25"].shift(1)) / d["ma25"].shift(1) * 100
            d["ma25_acceleration"] = d["ma25_velocity"].diff()
            
            d["relative_strength"] = d["Close"] / market_median_close
            d["rs_velocity"] = (d["relative_strength"] - d["relative_strength"].shift(5)) / d["relative_strength"].shift(5) * 100
            
            d["volatility_compression_ratio"] = d["Close"].rolling(20).std() / d["Close"].rolling(100).std()
            d["candle_mid_high"] = np.where(d["High"] > d["Low"], (d["Close"] - d["Low"]) / (d["High"] - d["Low"]) >= 0.5, False)
            d["is_positive_candle"] = d["Close"] > d["Open"]
            d["liquidity_ok"] = d["turnover_avg20_million"] >= TrackerConfig.min_daily_turnover_million
            
            # 追加フィルター：RSI(14) の算出
            delta = d["Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / np.where(loss > 0, loss, 1.0)
            d["rsi14"] = 100 - (100 / (1 + rs))
            
            # 【バグ修正】SyntaxErrorを回避するため、添字指定(subscript)への代入演算子(:=)の使用を廃止
            d["squeeze_duration"] = d["ma_squeeze_20d"].shift(1).groupby((~d["ma_squeeze_20d"].shift(1)).cumsum()).cumsum()
            
            d["sniper_motion"] = (
                (d["squeeze_duration"] >= 10) & 
                (d["volatility_compression_ratio"] <= 0.4) & 
                (d["vol_acceleration"] > 0) & 
                (d["volume_ratio_20"] >= 2.0) & 
                (d["ma25_acceleration"] > 0) & 
                (d["breakout_age"] == 1) & 
                (d["rs_velocity"] > 0) & 
                d["candle_mid_high"] & 
                d["is_positive_candle"] & 
                d["liquidity_ok"]
            )
            
            d["qualified_signal"] = d["sniper_motion"] & (d["rsi14"] >= 50.0) & (d["rsi14"] < 80.0)
            
            latest_row = d.iloc[-1]
            if bool(latest_row["qualified_signal"]):
                fundamentals = None
                if fund_path.exists():
                    try:
                        with open(fund_path, "r", encoding="utf-8") as f:
                            fundamentals = json.load(f)
                    except Exception:
                        pass
                
                signals.append({
                    "ticker": t,
                    "signal_date": d.index[-1].strftime("%Y-%m-%d"),
                    "signal_close": float(latest_row["Close"]),
                    "rsi14": round(float(latest_row["rsi14"]), 1),
                    "volume_ratio_20": round(float(latest_row["volume_ratio_20"]), 2),
                    "squeeze_ratio": round(float(latest_row["ma_congestion_width_pct"]), 2),
                    "sector": fundamentals.get("sector") if fundamentals else "不明",
                    "industry": fundamentals.get("industry") if fundamentals else "不明"
                })
        except Exception:
            continue
            
    return signals


def update_signal_lifecycle(today_signals: list[dict]):
    """
    シグナル台帳（CSV）に新規シグナルを追記し、既存のアクティブシグナルを毎日追跡更新します。
    """
    db_file = TrackerConfig.db_file
    prices_dir = TrackerConfig.cache_dir / "prices"
    
    # 既存の累積台帳ロード（なければ新規作成）
    if db_file.exists():
        df_db = pd.read_csv(db_file, encoding="utf-8-sig")
    else:
        df_db = pd.DataFrame(columns=[
            "ticker", "signal_date", "signal_close", "rsi14", "volume_ratio_20", "squeeze_ratio", 
            "sector", "industry", "status", "days_held", "current_close", "current_return_pct", 
            "max_high", "max_return_pct", "min_low", "min_return_pct", "max_drawdown_pct",
            "reached_20pct", "reached_30pct", "reached_50pct", "dropped_10pct"
        ])
        
    # 1. 本日の新規検知シグナルを台帳へ累積追加
    for sig in today_signals:
        is_duplicate = not df_db[(df_db["ticker"] == sig["ticker"]) & (df_db["signal_date"] == sig["signal_date"])].empty
        if is_duplicate:
            continue
            
        new_row = {
            "ticker": sig["ticker"],
            "signal_date": sig["signal_date"],
            "signal_close": sig["signal_close"],
            "rsi14": sig["rsi14"],
            "volume_ratio_20": sig["volume_ratio_20"],
            "squeeze_ratio": sig["squeeze_ratio"],
            "sector": sig["sector"],
            "industry": sig["industry"],
            "status": "active",
            "days_held": 0,
            "current_close": sig["signal_close"],
            "current_return_pct": 0.0,
            "max_high": sig["signal_close"],
            "max_return_pct": 0.0,
            "min_low": sig["signal_close"],
            "min_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "reached_20pct": False,
            "reached_30pct": False,
            "reached_50pct": False,
            "dropped_10pct": False
        }
        df_db = pd.concat([df_db, pd.DataFrame([new_row])], ignore_index=True)

    # 2. 追跡中（active）シグナルの日次アップデート処理
    active_mask = df_db["status"] == "active"
    for idx in df_db[active_mask].index:
        ticker = df_db.at[idx, "ticker"]
        sig_date_str = df_db.at[idx, "signal_date"]
        sig_close = float(df_db.at[idx, "signal_close"])
        
        price_path = prices_dir / f"{ticker}.csv"
        if not price_path.exists():
            continue
            
        try:
            d = pd.read_csv(price_path, index_col=0, parse_dates=True)
            d_after = d.loc[sig_date_str:]
            if d_after.empty:
                continue
                
            days_held = len(d_after) - 1
            current_row = d_after.iloc[-1]
            
            max_high = float(d_after["High"].max())
            min_low = float(d_after["Low"].min())
            current_close = float(current_row["Close"])
            
            max_return_pct = (max_high - sig_close) / sig_close * 100.0
            min_return_pct = (min_low - sig_close) / sig_close * 100.0
            current_return_pct = (current_close - sig_close) / sig_close * 100.0
            
            cum_max = d_after["High"].cummax()
            drawdowns = (d_after["Low"] - cum_max) / cum_max * 100.0
            max_drawdown_pct = float(drawdowns.min())
            
            reached_20pct = max_return_pct >= 20.0
            reached_30pct = max_return_pct >= 30.0
            reached_50pct = max_return_pct >= 50.0
            dropped_10pct = min_return_pct <= -10.0
            
            df_db.at[idx, "days_held"] = days_held
            df_db.at[idx, "current_close"] = current_close
            df_db.at[idx, "current_return_pct"] = round(current_return_pct, 2)
            df_db.at[idx, "max_high"] = max_high
            df_db.at[idx, "max_return_pct"] = round(max_return_pct, 2)
            df_db.at[idx, "min_low"] = min_low
            df_db.at[idx, "min_return_pct"] = round(min_return_pct, 2)
            df_db.at[idx, "max_drawdown_pct"] = round(max_drawdown_pct, 2)
            df_db.at[idx, "reached_20pct"] = reached_20pct
            df_db.at[idx, "reached_30pct"] = reached_30pct
            df_db.at[idx, "reached_50pct"] = reached_50pct
            df_db.at[idx, "dropped_10pct"] = dropped_10pct
            
            if days_held >= TrackerConfig.tracking_days_limit:
                df_db.at[idx, "status"] = "completed"
                
        except Exception:
            continue
            
    df_db.to_csv(db_file, index=False, encoding="utf-8-sig")
    print(f"シグナル自動追跡台帳を更新しました: 現在 {len(df_db)} 件のシグナルが累積されています。")


def generate_performance_report():
    """
    累積された台帳から、勝率や条件別分析を1秒で算出し「改善レポート（CSV）」として出力します。
    """
    db_file = TrackerConfig.db_file
    report_file = TrackerConfig.report_file
    
    if not db_file.exists():
        return
        
    df = pd.read_csv(db_file)
    if df.empty:
        return
        
    df["is_win"] = df["max_return_pct"] >= 15.0
    
    total_signals = len(df)
    win_rate = df["is_win"].mean() * 100
    median_gain = df["max_return_pct"].median()
    median_loss = df["min_return_pct"].median()
    avg_days = df["days_held"].mean()
    
    r_20 = df["reached_20pct"].mean() * 100
    r_30 = df["reached_30pct"].mean() * 100
    r_50 = df["reached_50pct"].mean() * 100
    d_10 = df["dropped_10pct"].mean() * 100
    
    report_rows = [
        {"Category": "Total Summary", "Condition": "All Signals", "Signals": total_signals, "Win Rate (%)": round(win_rate, 2), "Median Gain (%)": round(median_gain, 2), "Median Loss (%)": round(median_loss, 2), "Avg Days Held": round(avg_days, 1), "Reach 20% (%)": round(r_20, 2), "Reach 30% (%)": round(r_30, 2), "Reach 50% (%)": round(r_50, 2), "Drop -10% (%)": round(d_10, 2)}
    ]
    
    if "sector" in df.columns:
        for sector_name, grp in df.groupby("sector"):
            if len(grp) < 3:
                continue
            report_rows.append({
                "Category": "Sector Analysis",
                "Condition": sector_name,
                "Signals": len(grp),
                "Win Rate (%)": round(grp["is_win"].mean() * 100, 2),
                "Median Gain (%)": round(grp["max_return_pct"].median(), 2),
                "Median Loss (%)": round(grp["min_return_pct"].median(), 2),
                "Avg Days Held": round(grp["days_held"].mean(), 1),
                "Reach 20% (%)": round(grp["reached_20pct"].mean() * 100, 2),
                "Reach 30% (%)": round(grp["reached_30pct"].mean() * 100, 2),
                "Reach 50% (%)": round(grp["reached_50pct"].mean() * 100, 2),
                "Drop -10% (%)": round(grp["dropped_10pct"].mean() * 100, 2)
            })
            
    if "rsi14" in df.columns:
        for rsi_range, mask in [
            ("RSI 50-65 (安全型)", (df["rsi14"] >= 50) & (df["rsi14"] < 65)),
            ("RSI 65-80 (高推進型)", (df["rsi14"] >= 65) & (df["rsi14"] < 80))
        ]:
            grp = df[mask]
            if grp.empty:
                continue
            report_rows.append({
                "Category": "RSI Filter Analysis",
                "Condition": rsi_range,
                "Signals": len(grp),
                "Win Rate (%)": round(grp["is_win"].mean() * 100, 2),
                "Median Gain (%)": round(grp["max_return_pct"].median(), 2),
                "Median Loss (%)": round(grp["min_return_pct"].median(), 2),
                "Avg Days Held": round(grp["days_held"].mean(), 1),
                "Reach 20% (%)": round(grp["reached_20pct"].mean() * 100, 2),
                "Reach 30% (%)": round(grp["reached_30pct"].mean() * 100, 2),
                "Reach 50% (%)": round(grp["reached_50pct"].mean() * 100, 2),
                "Drop -10% (%)": round(grp["dropped_10pct"].mean() * 100, 2)
            })

    df_report = pd.DataFrame(report_rows)
    df_report.to_csv(report_file, index=False, encoding="utf-8-sig")
    print("勝率・改善案分析レポートの自動更新に成功しました。")


def notify_daily_signal(signals: list[dict], universe: pd.DataFrame):
    """
    毎日Actionsが自動実行され、合格銘柄があった際のスプレッドシート追記とメール送信です。
    """
    if not signals:
        print("\n本日、Sniper Motion の合格シグナルを検出した銘柄はありませんでした。")
        return
        
    name_map = dict(zip(universe["ticker"], universe["name"]))
    for sig in signals:
        sig["name"] = name_map.get(sig["ticker"], sig["ticker"])
        
    if WEBHOOK_URL:
        try:
            res = requests.post(WEBHOOK_URL, json={"signals": signals}, headers={"Content-Type": "application/json"})
            if res.status_code == 200:
                print("Googleスプレッドシートへの自動追記に成功しました。")
            else:
                print(f"スプレッドシート連携エラー: {res.text}")
        except Exception as e:
            print(f"スプレッドシート連携に失敗しました: {e}")
            
    if GMAIL_USER and GMAIL_PASS and NOTIFICATION_EMAIL:
        try:
            msg = MIMEMultipart()
            msg["From"] = GMAIL_USER
            msg["To"] = NOTIFICATION_EMAIL
            msg["Subject"] = f"【MOTION検知】Sniper Motion 心理転換初動銘柄 ({signals[0]['date']})"
            
            body = "静寂から需給の不均衡（最初の爆発）が起きた「Sniper Motion」の検知通知です。\n\n"
            body += "=========================================\n"
            for sig in signals:
                body += f"■ {sig['name']} ({sig['ticker']})\n"
                body += f"  ・終値: {sig['close']} 円\n"
                body += f"  ・RSI(14): {sig['rsi']} %\n"
                body += f"  ・セクター: {sig['sector']} / 業界: {sig['industry']}\n"
                body += "=========================================\n"
            body += "\n※スプレッドシートにも自動追記されました。翌朝の始値の動きを観察してください。\n"
            
            msg.attach(MIMEText(body, "plain", "utf-8"))
            
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(GMAIL_USER, GMAIL_PASS)
                server.sendmail(GMAIL_USER, [NOTIFICATION_EMAIL], msg.as_string())
            print("Sniper Motion 合格通知メールを送信しました。")
        except Exception as e:
            print(f"メール送信に失敗しました: {e}")


def main():
    if not UNIVERSE_CSV.exists():
        print("universe.csv が見つかりません。")
        return
        
    df_uni = pd.read_csv(UNIVERSE_CSV)
    df_uni["ticker"] = df_uni["ticker"].map(normalize_ticker)
    tickers = df_uni["ticker"].dropna().tolist()
    
    # 1. 本日分の株価データをマージアップデート（threads=FalseによるSQLite衝突回避）
    update_prices_daily_full(tickers)
    
    # 2. 相対強度のベンチマークとして「市場平均（本日の中央値Close）」を動的算出
    print("Relative Strength用：市場中央値時系列を算出中...")
    all_closes = {}
    for t in tickers:
        price_path = PRICES_DIR / f"{t}.csv"
        if price_path.exists():
            try:
                df_temp = pd.read_csv(price_path, index_col=0, parse_dates=True)
                all_closes[t] = df_temp["Close"].iloc[-100:]
            except Exception:
                continue
    df_all_closes = pd.DataFrame(all_closes)
    market_median_close = df_all_closes.median(axis=1).sort_index()
    
    # 3. 本日の新規「Sniper Motion」シグナルの検出
    today_signals = run_screener_for_today(tickers, market_median_close)
    
    # 4. 【ライフサイクル】シグナル台帳への追記と、アクティブシグナルの毎日追跡処理
    update_signal_lifecycle(today_signals)
    
    # 5. 【自己改善レポート】勝率や条件別分析を再集計
    generate_performance_report()
    
    # 6. スプレッドシート追記とメール通知の実行
    notify_daily_signal(today_signals, df_uni)


if __name__ == "__main__":
    main()
