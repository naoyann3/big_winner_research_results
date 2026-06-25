from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

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


def fetch_history(ticker: str, years: int) -> pd.DataFrame | None:
    period_days = max(365 * years + 120, 365)
    period = f"{period_days}d"
    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
    except Exception:
        return None

    if hist is None or hist.empty:
        return None

    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    need_cols = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in hist.columns for col in need_cols):
        return None

    hist = hist[need_cols].dropna().copy()
    if len(hist) < 220:
        return None
    return hist


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
    d["trend_building"] = (d["ma25"] > d["ma75"]) & (d["ma75"] > d["ma200"]) & (d["ma75_slope_pct"] > 0)
    d["breakout_52w"] = d["Close"] >= d["recent_high_52w"] * 0.98
    d["breakout_confirmed"] = d["breakout_52w"] & (d["volume_ratio_20"] >= 1.5)
    d["congestion_tight"] = d["ma_congestion_width_pct"] <= 8.0
    d["trend_building_entry"] = d["trend_building"] & ~d["trend_building"].shift(1).fillna(False)
    d["breakout_confirmed_entry"] = d["breakout_confirmed"] & ~d["breakout_confirmed"].shift(1).fillna(False)
    d["congestion_tight_entry"] = d["congestion_tight"] & ~d["congestion_tight"].shift(1).fillna(False)

    return d


def future_max_return(close: pd.Series, start_idx: int, forward_days: int) -> float | None:
    end_idx = min(start_idx + forward_days, len(close) - 1)
    if start_idx >= end_idx:
        return None
    future_max = float(close.iloc[start_idx + 1 : end_idx + 1].max())
    entry = float(close.iloc[start_idx])
    return (future_max - entry) / entry * 100.0


def find_events(hist: pd.DataFrame, fundamentals: dict, theme_context: dict, config: ResearchConfig) -> pd.DataFrame:
    features = calc_features(hist, config)
    rows = []
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
        ):
            if not entry_flag:
                continue

            fundamental_accel = (
                row.get("revenue_growth_pct") is not None
                and row.get("profit_margin_pct") is not None
                and row.get("roe_pct") is not None
            )

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
    summary = pd.DataFrame(rows).sort_values(["big_winner_rate", "signals"], ascending=[False, False]).reset_index(drop=True)

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
        extra_df = pd.DataFrame(extra_rows).sort_values(["big_winner_rate", "signals"], ascending=[False, False]).reset_index(drop=True)
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
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = ResearchConfig(
        years=args.years,
        forward_days=args.forward_days,
        big_winner_threshold_pct=args.big_winner_threshold_pct,
        fetch_timeout_sec=args.fetch_timeout_sec,
        max_tickers=args.max_tickers,
    )

    universe = load_universe(args.universe_csv, config.max_tickers)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    for i, row in universe.iterrows():
        ticker = row["ticker"]
        name = row.get("name", ticker)
        print(f"{i + 1}/{len(universe)} {ticker}")

        hist = run_with_timeout(fetch_history, config.fetch_timeout_sec, ticker, config.years)
        if hist is None:
            records.append({"ticker": ticker, "name": name, "hist": None, "fundamentals": None, "status": "no_history"})
            continue

        fundamentals = run_with_timeout(fetch_fundamentals, config.fetch_timeout_sec, ticker)
        if fundamentals is None:
            records.append({"ticker": ticker, "name": name, "hist": hist, "fundamentals": None, "status": "no_fundamentals"})
            continue

        records.append({"ticker": ticker, "name": name, "hist": hist, "fundamentals": fundamentals, "status": "ok"})

    theme_context = build_theme_context(records, config)

    all_events: list[pd.DataFrame] = []
    ticker_rows = []

    for rec in records:
        ticker = rec["ticker"]
        name = rec["name"]
        hist = rec["hist"]
        fundamentals = rec["fundamentals"]

        if hist is None:
            ticker_rows.append({"ticker": ticker, "name": name, "events": 0, "status": rec["status"]})
            continue
        if fundamentals is None:
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
                    "sector": fundamentals["sector"],
                    "industry": fundamentals["industry"],
                }
            )
            continue

        events.insert(0, "ticker", ticker)
        events.insert(1, "name", name)
        events["sector"] = fundamentals["sector"]
        events["industry"] = fundamentals["industry"]
        events["market_cap_billion"] = round((fundamentals["market_cap"] or 0.0) / 1_000_000_000, 3)
        events["revenue_growth_pct"] = fundamentals["revenue_growth_pct"]
        events["profit_margin_pct"] = fundamentals["profit_margin_pct"]
        events["roe_pct"] = fundamentals["roe_pct"]

        all_events.append(events)
        ticker_rows.append(
            {
                "ticker": ticker,
                "name": name,
                "events": int(len(events)),
                "big_winners": int(events["big_winner"].sum()),
                "big_winner_rate": float(events["big_winner"].mean()),
                "status": "ok",
                "sector": fundamentals["sector"],
                "industry": fundamentals["industry"],
            }
        )

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
