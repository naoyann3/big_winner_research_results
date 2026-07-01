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

    d["perfect_order"] = (d["Close"] > d["ma25"]) & (d["ma25"] > d["ma75"]) & (d["ma75"] > d["ma200"])
    d["perfect_order_forming"] = (d["ma25"] > d["ma75"]) & (d["ma75"] > d["ma200"]) & (d["ma75_slope_pct"] > 0)

    touch_tolerance = touch_tolerance_pct / 100.0
    low_touch = d["Low"] <= d["ma75"] * (1 + touch_tolerance)
    reclaim_close = d["Close"] >= d["ma75"]
    d["ma75_touch_reclaim"] = low_touch & reclaim_close

    wick_break = d["Low"] < d["ma75"]
    close_reclaim = d["Close"] >= d["ma75"]
    d["ma75_wick_reclaim"] = wick_break & close_reclaim

    d["signal_condition"] = d["perfect_order_forming"] & (d["ma75_touch_reclaim"] | d["ma75_wick_reclaim"])
    d["signal_day"] = d["signal_condition"] & ~d["signal_condition"].shift(1).fillna(False)

    return d


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
    target_price = entry_open * (1 + target_return_pct / 100.0)
    stop_price = entry_open * (1 + stop_loss_pct / 100.0)
    ma75_break_level = 1 - ma75_break_buffer_pct / 100.0

    window = hist.iloc[entry_idx : end_idx + 1].copy()
    window = window.reset_index(drop=True)

    target_hit_idx = None
    stop_hit_idx = None
    ambiguous_same_day = False

    for i, row in window.iterrows():
        high = float(row["High"])
        low = float(row["Low"])
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

    return {
        "signal_date": hist.index[signal_idx].date().isoformat(),
        "entry_date": hist.index[entry_idx].date().isoformat(),
        "exit_date": hist.index[exit_idx].date().isoformat(),
        "entry_open": entry_open,
        "target_price": target_price,
        "stop_price": stop_price,
        "outcome": outcome,
        "hit_day": hit_day,
        "pnl_pct": pnl_pct,
        "target_hit": outcome == "win",
        "stop_hit": outcome == "stop",
        "ambiguous_same_day": ambiguous_same_day,
        "days_held": (hit_day + 1) if hit_day is not None else (end_idx - entry_idx + 1),
    }


def simulate_ticker(
    ticker: str,
    name: str,
    config: BacktestConfig,
) -> tuple[list[dict], dict | None]:
    hist = fetch_history(ticker, config.years)
    if hist is None:
        return [], None

    hist = calc_indicators(hist, config.touch_tolerance_pct)
    valid = hist["signal_day"].fillna(False) & hist["ma75"].notna() & hist["ma200"].notna()
    signal_indices = np.flatnonzero(valid.to_numpy())

    event_rows: list[dict] = []
    for signal_idx in signal_indices:
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
                "signal_close": float(hist.iloc[signal_idx]["Close"]),
                "signal_ma25": float(hist.iloc[signal_idx]["ma25"]),
                "signal_ma75": float(hist.iloc[signal_idx]["ma75"]),
                "signal_ma200": float(hist.iloc[signal_idx]["ma200"]),
                "signal_volume_ratio_20": float(hist.iloc[signal_idx]["volume_ratio_20"]),
                "signal_turnover_million": float(hist.iloc[signal_idx]["turnover_million"]),
                "signal_ma75_slope_pct": float(hist.iloc[signal_idx]["ma75_slope_pct"]),
                "signal_ma200_slope_pct": float(hist.iloc[signal_idx]["ma200_slope_pct"]),
                "signal_condition": True,
                "perfect_order_forming": bool(hist.iloc[signal_idx]["perfect_order_forming"]),
                "perfect_order": bool(hist.iloc[signal_idx]["perfect_order"]),
                "ma75_touch_reclaim": bool(hist.iloc[signal_idx]["ma75_touch_reclaim"]),
                "ma75_wick_reclaim": bool(hist.iloc[signal_idx]["ma75_wick_reclaim"]),
            }
        )
        event_rows.append(row)

    if not event_rows:
        return [], None

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
    return event_rows, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest TSE Prime/Standard stocks for MA75 touch + next-open entry."
    )
    parser.add_argument("--universe-csv", type=Path, required=True, help="CSV with tickers and optional market_section")
    parser.add_argument("--output-dir", type=Path, default=Path("tse_ma75_backtest_results"))
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

    for i, row in universe.iterrows():
        ticker = row["ticker"]
        name = row.get("name", ticker)
        print(f"{i + 1}/{len(universe)} {ticker}")
        events, summary = simulate_ticker(ticker, name, config)
        all_events.extend(events)
        if summary is not None:
            all_summaries.append(summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    events_df = pd.DataFrame(all_events)
    summaries_df = pd.DataFrame(all_summaries)

    if not events_df.empty:
        events_df = events_df.sort_values(["ticker", "signal_date"]).reset_index(drop=True)
        events_df.to_csv(args.output_dir / "ma75_touch_event_log.csv", index=False, encoding="utf-8-sig")

    if not summaries_df.empty:
        summaries_df = summaries_df.sort_values(["win_rate", "signals"], ascending=[False, False]).reset_index(drop=True)
        summaries_df.to_csv(args.output_dir / "ma75_touch_ticker_summary.csv", index=False, encoding="utf-8-sig")

    total_signals = len(events_df) if not events_df.empty else 0
    total_wins = int((events_df["outcome"] == "win").sum()) if total_signals else 0
    total_stops = int((events_df["outcome"] == "stop").sum()) if total_signals else 0
    total_others = int((events_df["outcome"] == "other").sum()) if total_signals else 0
    win_rate = total_wins / total_signals if total_signals else 0.0
    stop_rate = total_stops / total_signals if total_signals else 0.0

    print("\n==== Overall Result ====")
    print(f"Universe tickers: {len(universe)}")
    print(f"Signals: {total_signals}")
    print(f"Wins: {total_wins} ({win_rate:.2%})")
    print(f"Stops: {total_stops} ({stop_rate:.2%})")
    print(f"Others: {total_others}")
    if total_signals:
        print(f"Average PnL: {events_df['pnl_pct'].mean():.2f}%")
        print(f"Median PnL: {events_df['pnl_pct'].median():.2f}%")
    print(f"Output dir: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
