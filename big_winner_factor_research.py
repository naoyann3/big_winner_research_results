from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import json
import time
import random

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class ResearchConfig:
    years: int = 5
    forward_days: int = 252
    big_winner_threshold_pct: float = 100.0
    breakout_lookback_days: int = 252
    congestion_lookback_days: int = 20
    min_revenue_growth_pct: float = 10.0
    min_profit_margin_pct: float = 10.0
    min_roe_pct: float = 10.0
    theme_top_sector_count: int = 6
    theme_top_industry_count: int = 12
    fetch_timeout_sec: int = 20
    max_tickers: int | None = None
    min_daily_turnover_million: float = 10.0  # 過去20日平均の1日あたり最低売買代金（百万円）
    cache_dir: Path = Path("data_cache")     # データキャッシュ保存先


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


def download_databank(universe: pd.DataFrame, config: ResearchConfig):
    """
    全銘柄の株価履歴とファンダメンタルズ情報をローカルにダウンロードして保存します。
    """
    cache_dir = config.cache_dir
    prices_dir = cache_dir / "prices"
    fund_dir = cache_dir / "fundamentals"
    
    prices_dir.mkdir(parents=True, exist_ok=True)
    fund_dir.mkdir(parents=True, exist_ok=True)
    
    tickers = universe["ticker"].tolist()
    
    print("\n--- 1. 株価履歴の一括ダウンロードを開始します ---")
    period_days = max(365 * config.years + 120, 365)
    period = f"{period_days}d"
    
    # 接続回数を極小化するため、300銘柄ずつのバルクバッチに分けてダウンロード
    batch_size = 300
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        print(f"株価バッチ取得中: {i+1}〜{min(i+batch_size, len(tickers))} / {len(tickers)}")
        
        try:
            # threads=False を指定することで、SQLiteの同時書き込みエラーを回避します
            data = yf.download(batch, period=period, interval="1d", group_by="ticker", auto_adjust=True, progress=False, threads=False)
            
            for t in batch:
                if isinstance(data.columns, pd.MultiIndex):
                    try:
                        t_data = data[t].dropna()
                        if not t_data.empty and len(t_data) >= 120:
                            t_data.to_csv(prices_dir / f"{t}.csv", index=True, encoding="utf-8-sig")
                    except KeyError:
                        pass
                else:
                    if not data.empty:
                        data.to_csv(prices_dir / f"{t}.csv", index=True, encoding="utf-8-sig")
        except Exception as e:
            print(f"  -> バッチ取得エラー (スキップして次へ): {e}")
            continue
                
    print("\n--- 2. ファンダメンタルズ情報のダウンロードを開始します ---")
    for idx, ticker in enumerate(tickers):
        cache_path = fund_dir / f"{ticker}.json"
        
        if cache_path.exists():
            continue
            
        print(f"[{idx+1}/{len(tickers)}] 取得中: {ticker}")
        
        fundamentals = run_with_timeout(fetch_fundamentals, config.fetch_timeout_sec, ticker)
        if fundamentals is not None:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(fundamentals, f, ensure_ascii=False, indent=4)
            time.sleep(random.uniform(1.5, 3.0))
        else:
            print(f"  -> {ticker} の財務データ取得に失敗（スキップ・次回再実行時に再試行）")

    print("\nデータバンクの構築処理がすべて完了しました。")


def price_return_pct(hist: pd.DataFrame, lookback: int) -> float | None:
    if len(hist) <= lookback:
        return None
    latest = float(hist["Close"].iloc[-1])
    past = float(hist["Close"].iloc[-1 - lookback])
    if past == 0:
        return None
    return (latest - past) / past * 100.0


def build_theme_context(records: list[dict], config: ResearchConfig) -> dict:
    sector_rows = []
    industry_rows = []
    for rec in records:
        hist = rec["hist"]
        fund = rec["fundamentals"]
        if hist is None or fund is None:
            continue
        r63 = price_return_pct(hist, 63)
        r126 = price_return_pct(hist, 126)
        if r63 is None or r126 is None:
            continue
        sector_rows.append(
            {
                "ticker": rec["ticker"],
                "sector": fund.get("sector"),
                "return_63d_pct": r63,
                "return_126d_pct": r126,
            }
        )
        industry_rows.append(
            {
                "ticker": rec["ticker"],
                "industry": fund.get("industry"),
                "return_63d_pct": r63,
                "return_126d_pct": r126,
            }
        )

    sector_df = pd.DataFrame(sector_rows)
    industry_df = pd.DataFrame(industry_rows)
    if sector_df.empty and industry_df.empty:
        return {"top_sectors": set(), "top_industries": set(), "sector_stats": pd.DataFrame(), "industry_stats": pd.DataFrame()}

    sector_stats = (
        sector_df.groupby("sector")
        .agg(
            sector_return_63d_median=("return_63d_pct", "median"),
            sector_return_126d_median=("return_126d_pct", "median"),
            sector_count=("ticker", "count"),
        )
        .reset_index()
    )
    sector_stats["theme_score"] = (
        sector_stats["sector_return_63d_median"].rank(pct=True) * 0.5
        + sector_stats["sector_return_126d_median"].rank(pct=True) * 0.5
    )
    top_sectors = set(
        sector_stats.sort_values(["theme_score", "sector_count"], ascending=[False, False])
        .head(config.theme_top_sector_count)["sector"]
        .dropna()
        .tolist()
    )

    if industry_df.empty:
        industry_stats = pd.DataFrame()
        top_industries = set()
    else:
        industry_stats = (
            industry_df.groupby("industry")
            .agg(
                industry_return_63d_median=("return_63d_pct", "median"),
                industry_return_126d_median=("return_126d_pct", "median"),
                industry_count=("ticker", "count"),
            )
            .reset_index()
        )
        industry_stats["theme_score"] = (
            industry_stats["industry_return_63d_median"].rank(pct=True) * 0.5
            + industry_stats["industry_return_126d_median"].rank(pct=True) * 0.5
        )
        top_industries = set(
            industry_stats.sort_values(["theme_score", "industry_count"], ascending=[False, False])
            .head(config.theme_top_industry_count)["industry"]
            .dropna()
            .tolist()
        )

    return {
        "top_sectors": top_sectors,
        "top_industries": top_industries,
        "sector_stats": sector_stats,
        "industry_stats": industry_stats,
    }


def calc_features(df: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    d = df.copy()
    d["ma25"] = d["Close"].rolling(25).mean()
    d["ma75"] = d["Close"].rolling(75).mean()
    d["ma200"] = d["Close"].rolling(200).mean()

    d["ma25_slope_pct"] = (d["ma25"] - d["ma25"].shift(5)) / d["ma25"].shift(5) * 100
    d["ma75_slope_pct"] = (d["ma75"] - d["ma75"].shift(5)) / d["ma75"].shift(5) * 100
    d["ma200_slope_pct"] = (d["ma200"] - d["ma200"].shift(5)) / d["ma200"].shift(5) * 100

    d["vol_avg20"] = d["Volume"].rolling(20).mean()
    d["volume_ratio_20"] = d["Volume"] / d["vol_avg20"]
    
    d["turnover_million"] = (d["Close"] * d["Volume"]) / 1_000_000
    d["turnover_avg20_million"] = d["turnover_million"].rolling(20).mean()
    
    d["liquidity_ok"] = d["turnover_avg20_million"] >= config.min_daily_turnover_million

    d["return_63d_pct"] = (d["Close"] - d["Close"].shift(63)) / d["Close"].shift(63) * 100
    d["return_126d_pct"] = (d["Close"] - d["Close"].shift(126)) / d["Close"].shift(126) * 100

    d["recent_high_52w"] = d["High"].rolling(config.breakout_lookback_days, min_periods=120).max()
    d["gap_to_52w_high_pct"] = (d["recent_high_52w"] - d["Close"]) / d["Close"] * 100
    d["close_vs_52w_high_pct"] = (d["Close"] - d["recent_high_52w"]) / d["recent_high_52w"] * 100

    d["ma_spread_25_75_pct"] = (d["ma25"] - d["ma75"]) / d["ma75"] * 100
    d["ma_spread_75_200_pct"] = (d["ma75"] - d["ma200"]) / d["ma200"] * 100
    d["ma_congestion_width_pct"] = (
        (d[["ma25", "ma75", "ma200"]].max(axis=1) - d[["ma25", "ma75", "ma200"]].min(axis=1))
        / d[["ma25", "ma75", "ma200"]].mean(axis=1)
        * 100
    )

    d["perfect_order"] = (d["Close"] > d["ma25"]) & (d["ma25"] > d["ma75"]) & (d["ma75"] > d["ma200"])
    
    d["trend_building"] = (d["ma25"] > d["ma75"]) & (d["ma75"] > d["ma200"]) & (d["ma75_slope_pct"] > 0) & d["liquidity_ok"]
    d["breakout_52w"] = (d["Close"] >= d["recent_high_52w"] * 0.98) & d["liquidity_ok"]
    d["breakout_confirmed"] = d["breakout_52w"] & (d["volume_ratio_20"] >= 1.5)
    d["congestion_tight"] = (d["ma_congestion_width_pct"] <= 8.0) & d["liquidity_ok"]

    # 1. 「MAの密集（スクイーズ）」の数値化
    d["ma_squeeze_20d"] = d["ma_congestion_width_pct"].rolling(20).max() <= 5.0

    # 2. 「売り枯れ（ドライアップ）」の検知
    vol_avg20_prior = d["Volume"].shift(1).rolling(20).mean()
    vol_avg5_prior = d["Volume"].shift(1).rolling(5).mean()
    d["dry_up"] = np.where(
        vol_avg20_prior > 0,
        vol_avg5_prior < (vol_avg20_prior * 0.8),
        False
    )

    # 3. 「直近の壁（レジスタンス）の突破」の検知
    d["recent_high_20d"] = d["High"].shift(1).rolling(20).max()
    d["breakout_20d"] = d["Close"] > d["recent_high_20d"]

    # 4. 「ローソクの質（引け値mid以上）」の判定
    d["candle_mid_high"] = np.where(
        d["High"] > d["Low"],
        (d["Close"] - d["Low"]) / (d["High"] - d["Low"]) >= 0.5,
        False
    )
    d["is_positive_candle"] = d["Close"] > d["Open"]

    # 5. 「初動スナイパー(sniper_breakout)」シグナルの統合
    d["sniper_breakout"] = (
        d["ma_squeeze_20d"] &
        d["dry_up"] &
        d["breakout_20d"] &
        (d["volume_ratio_20"] >= 1.5) &
        d["candle_mid_high"] &
        d["is_positive_candle"] &
        d["liquidity_ok"]
    )

    d["trend_building_entry"] = d["trend_building"] & ~d["trend_building"].shift(1).fillna(False)
    d["breakout_confirmed_entry"] = d["breakout_confirmed"] & ~d["breakout_confirmed"].shift(1).fillna(False)
    d["congestion_tight_entry"] = d["congestion_tight"] & ~d["congestion_tight"].shift(1).fillna(False)
    d["sniper_breakout_entry"] = d["sniper_breakout"] & ~d["sniper_breakout"].shift(1).fillna(False)

    return d


def future_max_return(close: pd.Series, start_idx: int, forward_days: int) -> float | None:
    end_idx = min(start_idx + forward_days, len(close) - 1)
    if start_idx >= end_idx:
        return None
    future_max = float(close.iloc[start_idx + 1 : end_idx + 1].max())
    entry = float(close.iloc[start_idx])
    return (future_max - entry) / entry * 100.0


def find_events(hist: pd.DataFrame, fundamentals: dict | None, theme_context: dict, config: ResearchConfig) -> pd.DataFrame:
    features = calc_features(hist, config)
    rows = []
    
    # ファンダメンタルズ情報がキャッシュに無くても(None)、検証を止めずにテクニカルのみで検証できるように修正
    if fundamentals is None:
        theme_tailwind = False
        fundamental_support = False
    else:
        theme_tailwind = (
            fundamentals.get("sector") in theme_context.get("top_sectors", set())
            or fundamentals.get("industry") in theme_context.get("top_industries", set())
        )
        fundamental_support = (
            fundamentals.get("revenue_growth_pct") is not None
            and fundamentals.get("profit_margin_pct") is not None
            and fundamentals.get("roe_pct") is not None
            and fundamentals.get("revenue_growth_pct") >= config.min_revenue_growth_pct
            and fundamentals.get("profit_margin_pct") >= config.min_profit_margin_pct
            and fundamentals.get("roe_pct") >= config.min_roe_pct
        )

    for idx in range(len(features)):
        row = features.iloc[idx]
        if pd.isna(row["ma200"]) or pd.isna(row["ma75"]) or pd.isna(row["ma25"]):
            continue

        for label, entry_flag in (
            ("trend_building", bool(row["trend_building_entry"])),
            ("breakout_confirmed", bool(row["breakout_confirmed_entry"])),
            ("congestion_tight", bool(row["congestion_tight_entry"])),
            ("sniper_breakout", bool(row["sniper_breakout_entry"])),
        ):
            if not entry_flag:
                continue

            fwd = future_max_return(features["Close"], idx, config.forward_days)
            if fwd is None:
                continue

            rows.append(
                {
                    "signal_date": features.index[idx].date().isoformat(),
                    "event_type": label,
                    "fundamental_support": bool(fundamental_support),
                    "theme_tailwind": bool(theme_tailwind),
                    "qualified_event": bool(fundamental_support) and bool(theme_tailwind),
                    "support_score": int(bool(fundamental_support)) + int(bool(theme_tailwind)),
                    "signal_close": float(row["Close"]),
                    "signal_open": float(row["Open"]),
                    "signal_volume_ratio_20": float(row["volume_ratio_20"]),
                    "signal_turnover_million": float(row["turnover_million"]),
                    "signal_ma25": float(row["ma25"]),
                    "signal_ma75": float(row["ma75"]),
                    "signal_ma200": float(row["ma200"]),
                    "ma25_slope_pct": float(row["ma25_slope_pct"]),
                    "ma75_slope_pct": float(row["ma75_slope_pct"]),
                    "ma200_slope_pct": float(row["ma200_slope_pct"]),
                    "ma_congestion_width_pct": float(row["ma_congestion_width_pct"]),
                    "close_vs_52w_high_pct": float(row["close_vs_52w_high_pct"]),
                    "gap_to_52w_high_pct": float(row["gap_to_52w_high_pct"]),
                    "perfect_order": bool(row["perfect_order"]),
                    "trend_building": bool(row["trend_building"]),
                    "breakout_52w": bool(row["breakout_52w"]),
                    "breakout_confirmed": bool(row["breakout_confirmed"]),
                    "congestion_tight": bool(row["congestion_tight"]),
                    "ma_squeeze_20d": bool(row.get("ma_squeeze_20d", False)),
                    "dry_up": bool(row.get("dry_up", False)),
                    "breakout_20d": bool(row.get("breakout_20d", False)),
                    "candle_mid_high": bool(row.get("candle_mid_high", False)),
                    "future_max_return_pct": fwd,
                    "big_winner": fwd >= config.big_winner_threshold_pct,
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
                "avg_volume_ratio_20": float(grp["signal_volume_ratio_20"].mean()),
                "avg_congestion_width_pct": float(grp["ma_congestion_width_pct"].mean()),
                "avg_close_vs_52w_high_pct": float(grp["close_vs_52w_high_pct"].mean()),
            }
        )
        
    summary = pd.DataFrame(rows).sort_values(["median_future_max_return_pct", "big_winner_rate"], ascending=[False, False]).reset_index(drop=True)

    extra_rows = []
    for label, mask in (
        ("qualified", events["qualified_event"]),
        ("fundamental_support_only", events["fundamental_support"]),
        ("theme_tailwind_only", events["theme_tailwind"]),
    ):
        grp = events[mask]
        if grp.empty:
            continue
        extra_rows.append(
            {
                "event_type": label,
                "signals": int(len(grp)),
                "big_winners": int(grp["big_winner"].sum()),
                "big_winner_rate": float(grp["big_winner"].mean()),
                "avg_future_max_return_pct": float(grp["future_max_return_pct"].mean()),
                "median_future_max_return_pct": float(grp["future_max_return_pct"].median()),
                "avg_volume_ratio_20": float(grp["signal_volume_ratio_20"].mean()),
                "avg_congestion_width_pct": float(grp["ma_congestion_width_pct"].mean()),
                "avg_close_vs_52w_high_pct": float(grp["close_vs_52w_high_pct"].mean()),
            }
        )

    if extra_rows:
        extra_df = pd.DataFrame(extra_rows).sort_values(["median_future_max_return_pct", "big_winner_rate"], ascending=[False, False]).reset_index(drop=True)
        summary = pd.concat([summary, extra_df], ignore_index=True)

    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research big-winner common factors.")
    parser.add_argument("--universe-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("big_winner_research_results"))
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--forward-days", type=int, default=252)
    parser.add_argument("--big-winner-threshold-pct", type=float, default=100.0)
    parser.add_argument("--fetch-timeout-sec", type=int, default=20)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--min-turnover-million", type=float, default=10.0,
                        help="Minimum 20-day average daily turnover in million JPY (default: 10.0)")
    parser.add_argument("--mode", type=str, choices=["download", "research", "all"], default="research",
                        help="Execution mode: 'download' (fetch & save data), 'research' (run simulation offline), 'all' (both)")
    parser.add_argument("--cache-dir", type=Path, default=Path("data_cache"),
                        help="Directory to store historical/fundamental caches")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = ResearchConfig(
        years=args.years,
        forward_days=args.forward_days,
        big_winner_threshold_pct=args.big_winner_threshold_pct,
        fetch_timeout_sec=args.fetch_timeout_sec,
        max_tickers=args.max_tickers,
        min_daily_turnover_million=args.min_turnover_million,
        cache_dir=args.cache_dir,
    )

    universe = load_universe(args.universe_csv, config.max_tickers)
    
    # 1. ダウンロードモード (または all) の実行
    if args.mode in ["download", "all"]:
        download_databank(universe, config)
        if args.mode == "download":
            return 

    # 2. リサーチモード (または all) の実行 (完全ローカル処理)
    print("\n--- ローカルデータバンクを用いた超高速検証を開始します ---")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    records: list[dict] = []
    prices_dir = config.cache_dir / "prices"
    fund_dir = config.cache_dir / "fundamentals"

    # --- 新規追加：読み込み進捗のリアルタイム表示 ---
    print("\nキャッシュファイルの読み込みを開始します...")
    for i, row in universe.iterrows():
        ticker = row["ticker"]
        name = row.get("name", ticker)
        
        # 100件ごとに進捗を画面に出力（フリーズではないことを明示します）
        if (i + 1) % 100 == 0 or (i + 1) == len(universe):
            print(f"  [ロード中] {i+1} / {len(universe)} 銘柄...")
        
        price_path = prices_dir / f"{ticker}.csv"
        fund_path = fund_dir / f"{ticker}.json"
        
        # 株価データさえローカルキャッシュにあれば、財務データがなくても検証を進められるよう変更
        if not price_path.exists():
            # キャッシュがないものは読み飛ばします
            continue
            
        try:
            # 日時情報を正しくロードするために index_col=0, parse_dates=True
            hist = pd.read_csv(price_path, index_col=0, parse_dates=True)
            
            # 財務データ（ファンダメンタルズ）の読み込み。存在しない場合は None とする
            fundamentals = None
            status_str = "no_fundamentals"
            if fund_path.exists():
                try:
                    with open(fund_path, "r", encoding="utf-8") as f:
                        fundamentals = json.load(f)
                    status_str = "ok"
                except Exception:
                    pass
            
            records.append({"ticker": ticker, "name": name, "hist": hist, "fundamentals": fundamentals, "status": status_str})
        except Exception:
            continue

    print(f"有効な株価キャッシュを読み込みました: {len(records)} / {len(universe)} 銘柄")

    print("\nテーマ情報の構築中...")
    theme_context = build_theme_context(records, config)

    all_events: list[pd.DataFrame] = []
    ticker_rows = []

    # --- 新規追加：シグナル計算のリアルタイム進捗表示 ---
    print("\nイベント検出と過去検証の計算を開始します...")
    for idx, rec in enumerate(records):
        ticker = rec["ticker"]
        name = rec["name"]
        hist = rec["hist"]
        fundamentals = rec["fundamentals"]

        if (idx + 1) % 100 == 0 or (idx + 1) == len(records):
            print(f"  [計算中] {idx+1} / {len(records)} 銘柄 ({ticker})...")

        # 財務データが無くても、株価履歴（hist）さえあればシグナル判定へ進む
        if hist is None:
            ticker_rows.append({"ticker": ticker, "name": name, "events": 0, "status": rec["status"]})
            continue

        events = find_events(hist, fundamentals, theme_context, config)
        if events.empty:
            ticker_rows.append(
                {
                    "ticker": ticker,
                    "name": name,
                    "events": 0,
                    "big_winners": 0,
                    "big_winner_rate": 0.0,
                    "status": "no_events",
                    "sector": fundamentals["sector"] if fundamentals else None,
                    "industry": fundamentals["industry"] if fundamentals else None,
                }
            )
            continue

        events.insert(0, "ticker", ticker)
        events.insert(1, "name", name)
        events["sector"] = fundamentals["sector"] if fundamentals else None
        events["industry"] = fundamentals["industry"] if fundamentals else None
        events["market_cap_billion"] = round((fundamentals["market_cap"] or 0.0) / 1_000_000_000, 3) if fundamentals else None
        events["revenue_growth_pct"] = fundamentals["revenue_growth_pct"] if fundamentals else None
        events["profit_margin_pct"] = fundamentals["profit_margin_pct"] if fundamentals else None
        events["roe_pct"] = fundamentals["roe_pct"] if fundamentals else None

        all_events.append(events)
        ticker_rows.append(
            {
                "ticker": ticker,
                "name": name,
                "events": int(len(events)),
                "big_winners": int(events["big_winner"].sum()),
                "big_winner_rate": float(events["big_winner"].mean()),
                "status": rec["status"],
                "sector": fundamentals["sector"] if fundamentals else None,
                "industry": fundamentals["industry"] if fundamentals else None,
            }
        )

    print("\n結果ファイルを書き出し中...")
    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    ticker_df = pd.DataFrame(ticker_rows)
    summary_df = summarize_events(events_df) if not events_df.empty else pd.DataFrame()

    if not events_df.empty:
        events_df.to_csv(args.output_dir / "big_winner_events.csv", index=False, encoding="utf-8-sig")
    if not ticker_df.empty:
        ticker_df.to_csv(args.output_dir / "big_winner_ticker_summary.csv", index=False, encoding="utf-8-sig")
    if not summary_df.empty:
        summary_df.to_csv(args.output_dir / "big_winner_event_summary.csv", index=False, encoding="utf-8-sig")

    if "sector_stats" in theme_context and isinstance(theme_context["sector_stats"], pd.DataFrame):
        theme_context["sector_stats"].to_csv(args.output_dir / "big_winner_theme_context.csv", index=False, encoding="utf-8-sig")
    if "industry_stats" in theme_context and isinstance(theme_context["industry_stats"], pd.DataFrame):
        theme_context["industry_stats"].to_csv(args.output_dir / "big_winner_industry_context.csv", index=False, encoding="utf-8-sig")

    print("\n==== Research Result ====")
    print(f"Universe tickers: {len(universe)}")
    print(f"Top sectors: {len(theme_context.get('top_sectors', set()))}")
    print(f"Top industries: {len(theme_context.get('top_industries', set()))}")
    print(f"Event rows: {len(events_df)}")
    if not events_df.empty:
        print(f"Big winners: {int(events_df['big_winner'].sum())}")
        print(f"Big winner rate: {events_df['big_winner'].mean():.2%}")
        print(f"Avg future max return: {events_df['future_max_return_pct'].mean():.2f}%")
        print(f"Median future max return: {events_df['future_max_return_pct'].median():.2f}%")
    print(f"Output dir: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
