from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class BacktestConfig:
    years: int = 3
    horizon_days: int = 20
    target_return_pct: float = 15.0
    stop_loss_pct: float = -5.0
    touch_tolerance_pct: float = 0.25
    ma75_break_buffer_pct: float = 0.50
    max_tickers: int | None = None
    min_turnover: float = 100_000_000
    min_market_cap: float = 30_000_000_000
    min_revenue_growth_pct: float = 5.0
    min_profit_margin_pct: float = 5.0
    min_roe_pct: float = 8.0
    max_52w_high_gap_pct: float = 20.0
    max_change_20d_pct: float = 25.0
    max_change_60d_pct: float = 80.0


def normalize_ticker(raw: str) -> str:
    ticker = str(raw).strip().upper()
    if not ticker:
        return ticker
    if "." not in ticker and not ticker.isdigit():
        ticker = f"{ticker}.T"
    return ticker


def parse_market_section(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip().lower()


def load_universe(path: Path, max_tickers: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError("Universe CSV must contain a 'ticker' column.")

    df = df.copy()
    df["ticker"] = df["ticker"].map(normalize_ticker)
    df = df[df["ticker"].astype(bool)]

    market_col = None
    for candidate in ("market_section", "market", "segment", "market_class"):
        if candidate in df.columns:
            market_col = candidate
            break

    if market_col is not None:
        market = df[market_col].map(parse_market_section)
        mask = market.str.contains("prime", na=False) | market.str.contains("standard", na=False)
        mask = mask | market.str.contains("プライム", na=False) | market.str.contains("スタンダード", na=False)
        df = df[mask]

    if max_tickers is not None:
        df = df.head(max_tickers)

    if "name" not in df.columns:
        df["name"] = df["ticker"]

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

    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in hist.columns for col in required):
        return None

    hist = hist[required].dropna().copy()
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


def calc_indicators(df: pd.DataFrame, touch_tolerance_pct: float) -> pd.DataFrame:
    d = df.copy()
    d["ma25"] = d["Close"].rolling(25).mean()
    d["ma75"] = d["Close"].rolling(75).mean()
    d["ma200"] = d["Close"].rolling(200).mean()

    d["ma75_slope_pct"] = (d["ma75"] - d["ma75"].shift(5)) / d["ma75"].shift(5) * 100
    d["ma200_slope_pct"] = (d["ma200"] - d["ma200"].shift(5)) / d["ma200"].shift(5) * 100
    d["vol_avg20"] = d["Volume"].rolling(20).mean()
    d["volume_ratio_20"] = d["Volume"] / d["vol_avg20"]
    d["turnover_million"] = (d["Close"] * d["Volume"]) / 1_000_000

    d["change_20d_pct"] = (d["Close"] - d["Close"].shift(20)) / d["Close"].shift(20) * 100
    d["change_60d_pct"] = (d["Close"] - d["Close"].shift(60)) / d["Close"].shift(60) * 100

    d["recent_high_252"] = d["High"].rolling(252, min_periods=120).max()
    d["gap_to_52w_high_pct"] = (d["recent_high_252"] - d["Close"]) / d["Close"] * 100

    d["perfect_order"] = (d["Close"] > d["ma25"]) & (d["ma25"] > d["ma75"]) & (d["ma75"] > d["ma200"])
    d["perfect_order_forming"] = (d["ma25"] > d["ma75"]) & (d["ma75"] > d["ma200"]) & (d["ma75_slope_pct"] > 0)

    touch_tolerance = touch_tolerance_pct / 100.0
    low_touch = d["Low"] <= d["ma75"] * (1 + touch_tolerance)
    reclaim_close = d["Close"] >= d["ma75"]
    wick_break = d["Low"] < d["ma75"]

    d["ma75_touch_reclaim"] = low_touch & reclaim_close
    d["ma75_wick_reclaim"] = wick_break & reclaim_close
    d["signal_condition"] = d["perfect_order_forming"] & (d["ma75_touch_reclaim"] | d["ma75_wick_reclaim"])
    d["signal_day"] = d["signal_condition"] & ~d["signal_condition"].shift(1).fillna(False)

    return d


def passes_long_term_filter(latest: pd.Series, f: dict, config: BacktestConfig) -> bool:
    if latest["turnover_million"] * 1_000_000 < config.min_turnover:
        return False
    if latest["Close"] < latest["ma75"]:
        return False
    if latest["ma25"] < latest["ma75"]:
        return False
    if pd.notna(latest["ma200"]) and latest["Close"] < latest["ma200"]:
        return False

    if latest["change_60d_pct"] < 0:
        return False
    if latest["gap_to_52w_high_pct"] > config.max_52w_high_gap_pct:
        return False
    if latest["change_20d_pct"] > config.max_change_20d_pct:
        return False
    if latest["change_60d_pct"] > config.max_change_60d_pct:
        return False

    if f["market_cap"] is None or f["market_cap"] < config.min_market_cap:
        return False
    if f["revenue_growth_pct"] is None or f["revenue_growth_pct"] < config.min_revenue_growth_pct:
        return False
    if f["profit_margin_pct"] is None or f["profit_margin_pct"] < config.min_profit_margin_pct:
        return False
    if f["roe_pct"] is None or f["roe_pct"] < config.min_roe_pct:
        return False

    return True


def simulate_one_signal(
    hist: pd.DataFrame,
    signal_idx: int,
    target_return_pct: float,
    stop_loss_pct: float,
    horizon_days: int,
    ma75_break_buffer_pct: float,
) -> dict | None:
    entry_idx = signal_idx + 1
    if entry_idx >= len(hist):
        return None

    end_idx = min(entry_idx + horizon_days - 1, len(hist) - 1)
    entry_open = float(hist.iloc[entry_idx]["Open"])
    signal_open = float(hist.iloc[signal_idx]["Open"])
    signal_high = float(hist.iloc[signal_idx]["High"])
    signal_low = float(hist.iloc[signal_idx]["Low"])
    signal_close = float(hist.iloc[signal_idx]["Close"])
    target_price = entry_open * (1 + target_return_pct / 100.0)
    stop_price = entry_open * (1 + stop_loss_pct / 100.0)
    ma75_break_level = 1 - ma75_break_buffer_pct / 100.0

    window = hist.iloc[entry_idx : end_idx + 1].copy().reset_index(drop=True)
    target_hit_idx = None
    stop_hit_idx = None
    ambiguous_same_day = False

    for i, row in window.iterrows():
        high = float(row["High"])
        close = float(row["Close"])
        ma75 = float(row["ma75"]) if pd.notna(row["ma75"]) else np.nan

        target_hit = high >= target_price
        stop_hit = (close <= stop_price) and pd.notna(ma75) and (close < ma75 * ma75_break_level)

        if target_hit and stop_hit:
            ambiguous_same_day = True
            stop_hit_idx = i
            target_hit_idx = i
            break
        if target_hit:
            target_hit_idx = i
            break
        if stop_hit:
            stop_hit_idx = i
            break

    if target_hit_idx is not None and (stop_hit_idx is None or target_hit_idx < stop_hit_idx):
        outcome = "win"
        hit_day = target_hit_idx
    elif stop_hit_idx is not None and (target_hit_idx is None or stop_hit_idx <= target_hit_idx):
        outcome = "stop"
        hit_day = stop_hit_idx
    else:
        outcome = "other"
        hit_day = None

    if ambiguous_same_day:
        outcome = "stop"
        hit_day = stop_hit_idx

    exit_idx = entry_idx + hit_day if hit_day is not None else end_idx
    exit_close = float(hist.iloc[exit_idx]["Close"])
    pnl_pct = (exit_close - entry_open) / entry_open * 100
    daily_range = max(signal_high - signal_low, 1e-9)
    close_location_pct = (signal_close - signal_low) / daily_range * 100.0
    upper_wick_pct = (signal_high - max(signal_open, signal_close)) / daily_range * 100.0
    lower_wick_pct = (min(signal_open, signal_close) - signal_low) / daily_range * 100.0

    return {
        "signal_date": hist.index[signal_idx].date().isoformat(),
        "entry_date": hist.index[entry_idx].date().isoformat(),
        "exit_date": hist.index[exit_idx].date().isoformat(),
        "entry_open": entry_open,
        "signal_open": signal_open,
        "signal_high": signal_high,
        "signal_low": signal_low,
        "signal_close": signal_close,
        "target_price": target_price,
        "stop_price": stop_price,
        "outcome": outcome,
        "hit_day": hit_day,
        "pnl_pct": pnl_pct,
        "signal_close_location_pct": close_location_pct,
        "signal_upper_wick_pct": upper_wick_pct,
        "signal_lower_wick_pct": lower_wick_pct,
        "target_hit": outcome == "win",
        "stop_hit": outcome == "stop",
        "ambiguous_same_day": ambiguous_same_day,
        "days_held": (hit_day + 1) if hit_day is not None else (end_idx - entry_idx + 1),
    }


def simulate_ticker(ticker: str, name: str, config: BacktestConfig) -> tuple[list[dict], dict | None, bool]:
    hist = fetch_history(ticker, config.years)
    if hist is None:
        return [], None, False

    fundamentals = fetch_fundamentals(ticker)
    if fundamentals is None:
        return [], None, False

    hist = calc_indicators(hist, config.touch_tolerance_pct)
    valid = hist["signal_day"].fillna(False) & hist["ma75"].notna() & hist["ma200"].notna()
    signal_indices = np.flatnonzero(valid.to_numpy())

    event_rows: list[dict] = []
    for signal_idx in signal_indices:
        latest = hist.iloc[signal_idx]
        if not passes_long_term_filter(latest, fundamentals, config):
            continue

        row = simulate_one_signal(
            hist=hist,
            signal_idx=int(signal_idx),
            target_return_pct=config.target_return_pct,
            stop_loss_pct=config.stop_loss_pct,
            horizon_days=config.horizon_days,
            ma75_break_buffer_pct=config.ma75_break_buffer_pct,
        )
        if row is None:
            continue

        row.update(
            {
                "ticker": ticker,
                "name": name,
                "signal_open": float(latest["Open"]),
                "signal_high": float(latest["High"]),
                "signal_low": float(latest["Low"]),
                "signal_close": float(latest["Close"]),
                "signal_ma25": float(latest["ma25"]),
                "signal_ma75": float(latest["ma75"]),
                "signal_ma200": float(latest["ma200"]),
                "signal_volume_ratio_20": float(latest["volume_ratio_20"]),
                "signal_turnover_million": float(latest["turnover_million"]),
                "signal_ma75_slope_pct": float(latest["ma75_slope_pct"]),
                "signal_ma200_slope_pct": float(latest["ma200_slope_pct"]),
                "signal_close_location_pct": round(
                    (float(latest["Close"]) - float(latest["Low"])) / max(float(latest["High"]) - float(latest["Low"]), 1e-9) * 100.0,
                    3,
                ),
                "signal_condition": True,
                "perfect_order_forming": bool(latest["perfect_order_forming"]),
                "perfect_order": bool(latest["perfect_order"]),
                "ma75_touch_reclaim": bool(latest["ma75_touch_reclaim"]),
                "ma75_wick_reclaim": bool(latest["ma75_wick_reclaim"]),
                "market_cap_billion": round(float(fundamentals["market_cap"]) / 1_000_000_000, 3),
                "revenue_growth_pct": float(fundamentals["revenue_growth_pct"]),
                "profit_margin_pct": float(fundamentals["profit_margin_pct"]),
                "roe_pct": float(fundamentals["roe_pct"]),
                "sector": fundamentals["sector"],
                "industry": fundamentals["industry"],
            }
        )
        event_rows.append(row)

    if not event_rows:
        return [], None, True

    per_ticker = pd.DataFrame(event_rows)
    summary = {
        "ticker": ticker,
        "name": name,
        "signals": int(len(per_ticker)),
        "wins": int((per_ticker["outcome"] == "win").sum()),
        "stops": int((per_ticker["outcome"] == "stop").sum()),
        "others": int((per_ticker["outcome"] == "other").sum()),
        "win_rate": float((per_ticker["outcome"] == "win").mean()),
        "stop_rate": float((per_ticker["outcome"] == "stop").mean()),
        "avg_pnl_pct": float(per_ticker["pnl_pct"].mean()),
        "median_pnl_pct": float(per_ticker["pnl_pct"].median()),
        "avg_days_held": float(per_ticker["days_held"].mean()),
    }
    return event_rows, summary, True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest TSE Prime/Standard stocks after long-term screener filters."
    )
    parser.add_argument("--universe-csv", type=Path, required=True, help="CSV with tickers and optional market_section")
    parser.add_argument("--output-dir", type=Path, default=Path("tse_ma75_backtest_filtered_results"))
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--horizon-days", type=int, default=20)
    parser.add_argument("--target-return-pct", type=float, default=15.0)
    parser.add_argument("--stop-loss-pct", type=float, default=-5.0)
    parser.add_argument("--touch-tolerance-pct", type=float, default=0.25)
    parser.add_argument("--ma75-break-buffer-pct", type=float, default=0.50)
    parser.add_argument("--max-tickers", type=int, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = BacktestConfig(
        years=args.years,
        horizon_days=args.horizon_days,
        target_return_pct=args.target_return_pct,
        stop_loss_pct=args.stop_loss_pct,
        touch_tolerance_pct=args.touch_tolerance_pct,
        ma75_break_buffer_pct=args.ma75_break_buffer_pct,
        max_tickers=args.max_tickers,
    )

    universe = load_universe(args.universe_csv, config.max_tickers)
    if universe.empty:
        raise ValueError("No tickers left after filtering the universe.")

    all_events: list[dict] = []
    all_summaries: list[dict] = []
    screened_count = 0
    valid_tickers = 0

    for i, row in universe.iterrows():
        ticker = row["ticker"]
        name = row.get("name", ticker)
        print(f"{i + 1}/{len(universe)} {ticker}")
        events, summary, passed_fundamentals = simulate_ticker(ticker, name, config)
        if passed_fundamentals:
            valid_tickers += 1
        if events:
            screened_count += 1
        all_events.extend(events)
        if summary is not None:
            all_summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    events_df = pd.DataFrame(all_events)
    summaries_df = pd.DataFrame(all_summaries)

    if not events_df.empty:
        events_df = events_df.sort_values(["ticker", "signal_date"]).reset_index(drop=True)
        events_df.to_csv(args.output_dir / "ma75_touch_event_log_filtered.csv", index=False, encoding="utf-8-sig")

    if not summaries_df.empty:
        summaries_df = summaries_df.sort_values(["win_rate", "signals"], ascending=[False, False]).reset_index(drop=True)
        summaries_df.to_csv(args.output_dir / "ma75_touch_ticker_summary_filtered.csv", index=False, encoding="utf-8-sig")

    if not events_df.empty:
        matrix_df = build_condition_matrix(events_df)
        matrix_df.to_csv(args.output_dir / "ma75_touch_condition_matrix_filtered.csv", index=False, encoding="utf-8-sig")
        rule_df = build_rule_summaries(events_df)
        rule_df.to_csv(args.output_dir / "ma75_touch_rule_summary_filtered.csv", index=False, encoding="utf-8-sig")

    total_signals = len(events_df) if not events_df.empty else 0
    total_wins = int((events_df["outcome"] == "win").sum()) if total_signals else 0
    total_stops = int((events_df["outcome"] == "stop").sum()) if total_signals else 0
    total_others = int((events_df["outcome"] == "other").sum()) if total_signals else 0
    win_rate = total_wins / total_signals if total_signals else 0.0
    stop_rate = total_stops / total_signals if total_signals else 0.0

    print("\n==== Filtered Overall Result ====")
    print(f"Universe tickers: {len(universe)}")
    print(f"Fundamentally valid tickers: {valid_tickers}")
    print(f"Signal-bearing tickers: {screened_count}")
    print(f"Signals: {total_signals}")
    print(f"Wins: {total_wins} ({win_rate:.2%})")
    print(f"Stops: {total_stops} ({stop_rate:.2%})")
    print(f"Others: {total_others}")
    if total_signals:
        print(f"Average PnL: {events_df['pnl_pct'].mean():.2f}%")
        print(f"Median PnL: {events_df['pnl_pct'].median():.2f}%")
    print(f"Output dir: {args.output_dir.resolve()}")


def build_condition_matrix(events_df: pd.DataFrame) -> pd.DataFrame:
    df = events_df.copy()

    df["volume_bin"] = pd.cut(
        df["signal_volume_ratio_20"],
        bins=[-np.inf, 1.0, 1.2, 1.5, np.inf],
        labels=["<1.0", "1.0-1.2", "1.2-1.5", ">=1.5"],
        right=False,
    )
    df["close_loc_bin"] = pd.cut(
        df["signal_close_location_pct"],
        bins=[-np.inf, 33.0, 66.0, np.inf],
        labels=["low", "mid", "high"],
        right=False,
    )

    rows = []
    for (vol_bin, close_bin), grp in df.groupby(["volume_bin", "close_loc_bin"], dropna=False):
        if grp.empty:
            continue
        rows.append(
            {
                "volume_bin": str(vol_bin),
                "close_loc_bin": str(close_bin),
                "signals": int(len(grp)),
                "wins": int((grp["outcome"] == "win").sum()),
                "stops": int((grp["outcome"] == "stop").sum()),
                "others": int((grp["outcome"] == "other").sum()),
                "win_rate": float((grp["outcome"] == "win").mean()),
                "stop_rate": float((grp["outcome"] == "stop").mean()),
                "avg_pnl_pct": float(grp["pnl_pct"].mean()),
                "median_pnl_pct": float(grp["pnl_pct"].median()),
                "avg_days_held": float(grp["days_held"].mean()),
            }
        )

    matrix = pd.DataFrame(rows)
    if matrix.empty:
        return matrix

    return matrix.sort_values(["volume_bin", "close_loc_bin"]).reset_index(drop=True)


def build_rule_summaries(events_df: pd.DataFrame) -> pd.DataFrame:
    rules = [
        ("all", events_df),
        ("volume>=1.5", events_df[events_df["signal_volume_ratio_20"] >= 1.5]),
        (
            "volume>=1.5 & close_loc>=33",
            events_df[
                (events_df["signal_volume_ratio_20"] >= 1.5)
                & (events_df["signal_close_location_pct"] >= 33.0)
            ],
        ),
        (
            "volume>=1.5 & close_loc>=66",
            events_df[
                (events_df["signal_volume_ratio_20"] >= 1.5)
                & (events_df["signal_close_location_pct"] >= 66.0)
            ],
        ),
        (
            "volume>=1.5 & close_loc<33",
            events_df[
                (events_df["signal_volume_ratio_20"] >= 1.5)
                & (events_df["signal_close_location_pct"] < 33.0)
            ],
        ),
        (
            "volume>=1.5 & close_loc mid",
            events_df[
                (events_df["signal_volume_ratio_20"] >= 1.5)
                & (events_df["signal_close_location_pct"] >= 33.0)
                & (events_df["signal_close_location_pct"] < 66.0)
            ],
        ),
    ]

    rows = []
    for label, grp in rules:
        if grp.empty:
            continue
        rows.append(
            {
                "rule": label,
                "signals": int(len(grp)),
                "wins": int((grp["outcome"] == "win").sum()),
                "stops": int((grp["outcome"] == "stop").sum()),
                "others": int((grp["outcome"] == "other").sum()),
                "win_rate": float((grp["outcome"] == "win").mean()),
                "stop_rate": float((grp["outcome"] == "stop").mean()),
                "avg_pnl_pct": float(grp["pnl_pct"].mean()),
                "median_pnl_pct": float(grp["pnl_pct"].median()),
                "avg_days_held": float(grp["days_held"].mean()),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(["win_rate", "signals"], ascending=[False, False]).reset_index(drop=True)


if __name__ == "__main__":
    main()
