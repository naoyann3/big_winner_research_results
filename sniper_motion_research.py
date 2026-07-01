from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import json
import time
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class MotionConfig:
    years: int = 5
    forward_days: int = 252
    big_winner_threshold_pct: float = 100.0
    fetch_timeout_sec: int = 20
    max_tickers: int | None = None
    min_daily_turnover_million: float = 10.0  # 最低1,000万円以上の売買代金
    cache_dir: Path = Path("data_cache")     # 既存のデータキャッシュを共通利用
    output_dir: Path = Path("big_winner_research_results")


def normalize_ticker(raw: str) -> str:
    ticker = str(raw).strip().upper()
    if not ticker:
        return ticker
    if "." not in ticker and not ticker.isdigit():
        ticker = f"{ticker}.T"
    return ticker


def run_with_timeout(func, timeout_sec: int, *args, **kwargs):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout_sec)
    except FuturesTimeoutError:
        future.cancel()
        return None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def load_universe(path: Path, max_tickers: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError("Universe CSV must contain a 'ticker' column.")

    df = df.copy()
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df = df[df["ticker"].astype(bool)]
    if "name" not in df.columns:
        df["name"] = df["ticker"]
    if max_tickers is not None:
        df = df.head(max_tickers)
    return df.reset_index(drop=True)


def update_prices_daily(tickers: list[str], config: MotionConfig):
    """
    株価データの差分/新規構築を判断して実行します（threads=FalseでSQLite競合を回避）。
    """
    prices_dir = config.cache_dir / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    
    existing_tickers = []
    new_tickers = []
    
    for t in tickers:
        if (prices_dir / f"{t}.csv").exists():
            existing_tickers.append(t)
        else:
            new_tickers.append(t)
            
    # ① キャッシュが存在しない銘柄（新規実行時）：5年分（1945日）をバルク取得
    if new_tickers:
        print(f"\n新規銘柄（キャッシュなし）の5年分株価ダウンロードを開始します... (対象: {len(new_tickers)} 銘柄)")
        period_days = max(365 * config.years + 120, 365)
        period_new = f"{period_days}d"
        _download_and_merge(new_tickers, period_new, prices_dir, is_new=True)
        
    # ② キャッシュがすでに存在する銘柄：最新5日分を落として末尾に差分マージ
    if existing_tickers:
        print(f"\n既存銘柄の差分更新（最新5日分）を開始します... (対象: {len(existing_tickers)} 銘柄)")
        _download_and_merge(existing_tickers, "5d", prices_dir, is_new=False)


def _download_and_merge(tickers: list[str], period: str, prices_dir: Path, is_new: bool):
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
            print(f"  ダウンロードエラー: {e}")
            continue


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / np.where(loss > 0, loss, 1.0)
    return 100 - (100 / (1 + rs))


# ==========================================
# --- 新規：動画（時間変化）特徴量の計算 ---
# ==========================================

def calc_motion_features(df: pd.DataFrame, market_median_close: pd.Series | None, config: MotionConfig) -> pd.DataFrame:
    d = df.copy()
    
    # 1. 基本となる移動平均線
    d["ma25"] = d["Close"].rolling(25).mean()
    d["ma75"] = d["Close"].rolling(75).mean()
    d["ma200"] = d["Close"].rolling(200).mean()
    
    # 2. 売買代金と最低流動性フィルター
    d["turnover_million"] = (d["Close"] * d["Volume"]) / 1_000_000
    d["turnover_avg20_million"] = d["turnover_million"].rolling(20).mean()
    d["liquidity_ok"] = d["turnover_avg20_million"] >= config.min_daily_turnover_million
    
    # 3. 【出来高の時間ダイナミクス】
    # 出来高の速度（5日間の線形スロープ）
    d["vol_velocity"] = d["Volume"].rolling(5).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 5 else 0.0, raw=True
    )
    # 出来高の加速度（速度の差分変化）
    d["vol_acceleration"] = d["vol_velocity"].diff()
    # 何営業日ぶりの異常出来高か（過去120日間の最大出来高を更新したかどうかのルックバック期間）
    d["vol_avg20"] = d["Volume"].rolling(20).mean()
    d["volume_ratio_20"] = d["Volume"] / d["vol_avg20"]
    d["days_since_vol_spike"] = d["volume_ratio_20"].rolling(120).apply(
        lambda x: len(x) - 1 - np.argmax(x) if len(x) > 0 else 0, raw=True
    )
    
    # 4. 【移動平均線の速度（1次導関数）と加速度（2次導関数）】
    # MA25の傾きの変化（速度と加速度）
    d["ma25_velocity"] = (d["ma25"] - d["ma25"].shift(1)) / d["ma25"].shift(1) * 100
    d["ma25_acceleration"] = d["ma25_velocity"].diff()
    # MA75の傾きの変化（速度と加速度）
    d["ma75_velocity"] = (d["ma75"] - d["ma75"].shift(1)) / d["ma75"].shift(1) * 100
    d["ma75_acceleration"] = d["ma75_velocity"].diff()
    
    # 5. 【スクイーズ（収縮）の継続期間とボラティリティ収縮率】
    # MAの最大乖離幅
    d["ma_congestion_width_pct"] = (
        (d[["ma25", "ma75", "ma200"]].max(axis=1) - d[["ma25", "ma75", "ma200"]].min(axis=1))
        / d[["ma25", "ma75", "ma200"]].mean(axis=1) * 100
    )
    # 乖離率5%以下のスクイーズ状態
    d["is_squeezed"] = d["ma_congestion_width_pct"] <= 5.0
    # スクイーズが何営業日連続して継続しているか（スクイーズ期間の年齢）
    d["squeeze_duration"] = d["is_squeezed"].groupby((~d["is_squeezed"]).cumsum()).cumsum()
    # ボラティリティ収縮率（過去100日STDに対する直近20日STDの比率）
    d["volatility_compression_ratio"] = d["Close"].rolling(20).std() / d["Close"].rolling(100).std()
    
    # 6. 【ブレイクアウト年齢の算出】
    d["recent_high_20d"] = d["High"].shift(1).rolling(20).max()
    d["is_breakout"] = d["Close"] > d["recent_high_20d"]
    # ブレイクアウトが何営業日継続しているか（ブレイク年齢）
    d["breakout_age"] = d["is_breakout"].groupby((~d["is_breakout"]).cumsum()).cumsum()
    
    # 7. 【動的・相対強度（Relative Strength: RS）】
    # 市場全体の中央値と比較した個別の強さ
    if market_median_close is not None:
        d["relative_strength"] = d["Close"] / market_median_close
        # 相対強度の改善速度（スロープ）
        d["rs_velocity"] = (d["relative_strength"] - d["relative_strength"].shift(5)) / d["relative_strength"].shift(5) * 100
    else:
        d["relative_strength"] = 1.0
        d["rs_velocity"] = 0.0
        
    d["rsi14"] = calculate_rsi(d["Close"], 14)
    d["is_positive_candle"] = d["Close"] > d["Open"]
    
    # 1日の実幅の中央値以上で引けたか
    d["candle_mid_high"] = np.where(d["High"] > d["Low"], (d["Close"] - d["Low"]) / (d["High"] - d["Low"]) >= 0.5, False)

    # ==========================================
    # --- 「Sniper Motion (静止状態から動き始める瞬間)」シグナルの定義 ---
    # ==========================================
    # 1. 10営業日以上、極限のスクイーズ（エネルギーの蓄積）が続いていたこと
    # 2. ボラティリティが過去の長期に比べて著しく収縮していること
    # 3. 出来高の速度・加速度が急増していること（最初の爆発）
    # 4. MA25の加速度がプラスに転じ、直近20日高値を上抜けた「ブレイク1日目」であること
    # 5. 相対強度（RS）の改善速度が上向き（プラス）に転じたこと
    
    d["sniper_motion"] = (
        (d["squeeze_duration"].shift(1) >= 10) &      # 前日まで10日以上スクイーズが継続
        (d["volatility_compression_ratio"] <= 0.4) &  # ボラティリティが長期平均の4割以下に圧縮
        (d["vol_acceleration"] > 0) &                 # 出来高の増加スピードが加速
        (d["volume_ratio_20"] >= 2.0) &               # 出来高が平均の2倍以上（爆発）
        (d["ma25_acceleration"] > 0) &                # 移動平均の上向きスピードが加速
        (d["breakout_age"] == 1) &                    # 20日高値ブレイクのちょうど「1日目」
        (d["rs_velocity"] > 0) &                      # 市場平均に対する強度が改善中
        d["candle_mid_high"] &                        # 上ヒゲを否定して引ける
        d["is_positive_candle"] &                     # 陽線
        d["liquidity_ok"]                             # 最低売買代金をクリア
    )

    return d

# ==========================================


def future_max_return(close: pd.Series, start_idx: int, forward_days: int) -> float | None:
    end_idx = min(start_idx + forward_days, len(close) - 1)
    if start_idx >= end_idx:
        return None
    future_max = float(close.iloc[start_idx + 1 : end_idx + 1].max())
    entry = float(close.iloc[start_idx])
    return (future_max - entry) / entry * 100.0


def future_min_return(close: pd.Series, start_idx: int, forward_days: int) -> float | None:
    end_idx = min(start_idx + forward_days, len(close) - 1)
    if start_idx >= end_idx:
        return None
    future_min = float(close.iloc[start_idx + 1 : end_idx + 1].min())
    entry = float(close.iloc[start_idx])
    return (future_min - entry) / entry * 100.0


def find_motion_events(hist: pd.DataFrame, fundamentals: dict | None, market_median_close: pd.Series | None, config: MotionConfig) -> pd.DataFrame:
    features = calc_motion_features(hist, market_median_close, config)
    rows = []
    
    for idx in range(len(features)):
        row = features.iloc[idx]
        if pd.isna(row["ma200"]) or pd.isna(row["ma75"]) or pd.isna(row["ma25"]):
            continue

        # 本日にシグナルが発生しているか
        if bool(row["sniper_motion"]):
            fwd_max = future_max_return(features["Close"], idx, config.forward_days)
            fwd_min = future_min_return(features["Close"], idx, config.forward_days)
            
            if fwd_max is None or fwd_min is None:
                continue

            rows.append(
                {
                    "signal_date": features.index[idx].date().isoformat(),
                    "event_type": "sniper_motion",
                    "signal_close": float(row["Close"]),
                    "signal_open": float(row["Open"]),
                    "signal_volume_ratio_20": float(row["volume_ratio_20"]),
                    "signal_turnover_million": float(row["turnover_million"]),
                    "ma_congestion_width_pct": float(row["ma_congestion_width_pct"]),
                    "volatility_compression_ratio": float(row["volatility_compression_ratio"]),
                    "vol_acceleration": float(row["vol_acceleration"]),
                    "ma25_acceleration": float(row["ma25_acceleration"]),
                    "relative_strength": float(row["relative_strength"]),
                    "rs_velocity": float(row["rs_velocity"]),
                    "future_max_return_pct": fwd_max,
                    "future_min_return_pct": fwd_min,
                    "big_winner": fwd_max >= config.big_winner_threshold_pct,
                }
            )

    return pd.DataFrame(rows)


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events

    rows = []
    for event_type, grp in events.groupby("event_type"):
        rows.append(
            {
                "event_type": event_type,
                "signals": int(len(grp)),
                "big_winners": int(grp["big_winner"].sum()),
                "big_winner_rate": float(grp["big_winner"].mean()),
                "avg_future_max_return_pct": float(grp["future_max_return_pct"].mean()),
                "median_future_max_return_pct": float(grp["future_max_return_pct"].median()),
                "avg_future_min_return_pct": float(grp["future_min_return_pct"].mean()),
                "median_future_min_return_pct": float(grp["future_min_return_pct"].median()),
                "avg_volume_ratio_20": float(grp["signal_volume_ratio_20"].mean()),
                "avg_congestion_width_pct": float(grp["ma_congestion_width_pct"].mean()),
            }
        )
    return pd.DataFrame(rows)


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
        
    # 1. Googleスプレッドシートへの追記 (GAS Webhook)
    if WEBHOOK_URL:
        try:
            res = requests.post(WEBHOOK_URL, json={"signals": signals}, headers={"Content-Type": "application/json"})
            if res.status_code == 200:
                print("Googleスプレッドシートへの自動追記に成功しました。")
            else:
                print(f"スプレッドシート連携エラー: {res.text}")
        except Exception as e:
            print(f"スプレッドシート連携に失敗しました: {e}")
            
    # 2. メール通知
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research Sniper Motion - market psychological shifts.")
    parser.add_argument("--universe-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("big_winner_research_results"))
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--forward-days", type=int, default=252)
    parser.add_argument("--big-winner-threshold-pct", type=float, default=100.0)
    parser.add_argument("--fetch-timeout-sec", type=int, default=20)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--min-turnover-million", type=float, default=10.0,
                        help="Minimum 20-day average daily turnover in million JPY (default: 10.0)")
    parser.add_argument("--mode", type=str, choices=["download", "research", "daily"], default="research",
                        help="Execution mode: 'download' (update cache), 'research' (run simulator), 'daily' (run daily screener for Actions)")
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = MotionConfig(
        years=args.years,
        forward_days=args.forward_days,
        big_winner_threshold_pct=args.big_winner_threshold_pct,
        max_tickers=args.max_tickers,
        min_daily_turnover_million=args.min_turnover_million,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
    )

    universe = load_universe(args.universe_csv, config.max_tickers)
    
    # 1. キャッシュアップデート（手動、または自動）
    if args.mode == "download":
        update_prices_daily(universe["ticker"].tolist(), config)
        return

    # 2. データのロード
    print("\nローカルキャッシュデータから検証データをロード中...")
    records: list[dict] = []
    prices_dir = config.cache_dir / "prices"
    fund_dir = config.cache_dir / "fundamentals"

    for i, row in universe.iterrows():
        ticker = row["ticker"]
        name = row.get("name", ticker)
        
        price_path = prices_dir / f"{ticker}.csv"
        fund_path = fund_dir / f"{ticker}.json"
        
        if (i + 1) % 500 == 0 or (i + 1) == len(universe):
            print(f"  [ロード中] {i+1} / {len(universe)} 銘柄...")
            
        if not price_path.exists():
            continue
            
        try:
            hist = pd.read_csv(price_path, index_col=0, parse_dates=True)
            fundamentals = None
            if fund_path.exists():
                try:
                    with open(fund_path, "r", encoding="utf-8") as f:
                        fundamentals = json.load(f)
                except Exception:
                    pass
            
            records.append({"ticker": ticker, "name": name, "hist": hist, "fundamentals": fundamentals})
        except Exception:
            continue

    print(f"有効な株価キャッシュを読み込みました: {len(records)} / {len(universe)} 銘柄")

    # 3. 【最重要】市場全体の日付ごとの中央値（ベンチマーク）を動的に生成
    print("\n市場全体の中央値時系列（Relative Strength用ベンチマーク）を算出中...")
    all_closes = {}
    for rec in records:
        all_closes[rec["ticker"]] = rec["hist"]["Close"]
    df_all_closes = pd.DataFrame(all_closes)
    market_median_close = df_all_closes.median(axis=1).sort_index()

    # 4. モードに沿ったメイン処理
    if args.mode == "research":
        # 過去検証（シミュレーション）モード
        print("\n--- Sniper Motion 過去シミュレーションの計算を開始します ---")
        all_events = []
        
        for idx, rec in enumerate(records):
            ticker = rec["ticker"]
            name = rec["name"]
            hist = rec["hist"]
            fundamentals = rec["fundamentals"]
            
            if (idx + 1) % 500 == 0 or (idx + 1) == len(records):
                print(f"  [計算中] {idx+1} / {len(records)} 銘柄 ({ticker})...")
                
            events = find_motion_events(hist, fundamentals, market_median_close, config)
            if events.empty:
                continue
                
            events.insert(0, "ticker", ticker)
            events.insert(1, "name", name)
            all_events.append(events)
            
        events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
        summary_df = summarize_events(events_df) if not events_df.empty else pd.DataFrame()
        
        # 結果の保存
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if not events_df.empty:
            events_df.to_csv(config.output_dir / "sniper_motion_events.csv", index=False, encoding="utf-8-sig")
            summary_df.to_csv(config.output_dir / "sniper_motion_event_summary.csv", index=False, encoding="utf-8-sig")
            
        print("\n==== Simulation Result ====")
        print(f"Total events found: {len(events_df)}")
        if not events_df.empty:
            print(f"Big winner rate: {events_df['big_winner'].mean():.2%}")
            print(f"Median max return: {events_df['future_max_return_pct'].median():.2f}%")
            print(f"Median min return (Risk): {events_df['future_min_return_pct'].median():.2f}%")
        print(f"Results saved in: {config.output_dir.resolve()}")

    elif args.mode == "daily":
        # 毎日 Actions で動かすスクリーナーモード
        print("\n--- Sniper Motion 本日シグナルの検知を開始します ---")
        
        # 最新日データをアップデート
        update_prices_daily(universe["ticker"].tolist(), config)
        
        daily_signals = []
        for rec in records:
            ticker = rec["ticker"]
            name = rec["name"]
            hist = rec["hist"]
            fundamentals = rec["fundamentals"]
            
            # 各個別銘柄のMotion特徴量を最新状態（今日）で計算
            d = calc_motion_features(hist, market_median_close, config)
            if len(d) < 220:
                continue
                
            latest_row = d.iloc[-1]
            if bool(latest_row["sniper_motion"]):
                daily_signals.append({
                    "date": d.index[-1].strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "name": name,
                    "close": float(latest_row["Close"]),
                    "rsi": round(float(latest_row["rsi14"]), 1),
                    "sector": fundamentals.get("sector") if fundamentals else "不明",
                    "industry": fundamentals.get("industry") if fundamentals else "不明"
                })
                
        # 通知の実行
        notify_daily_signal(daily_signals, universe)


if __name__ == "__main__":
    main()
