"""
Fetch forward-adjusted daily OHLCV data (including turnover rate) for
all A-share markets (Shanghai, Shenzhen, Beijing) using pytdx.

The script:
1. Selects the fastest TDX quote server.
2. Enumerates all securities across SH/SZ/BJ.
3. Downloads full daily history per code.
4. Applies 前复权 (forward adjustment) so the latest prices stay
   unchanged while earlier prices are adjusted and can be negative
   after deep dividends.
5. Computes turnover rate using circulating shares from finance info.
6. Writes one CSV per security under the output directory.

Run:
    python download_all_daily_qfq.py --out data/daily_qfq

Note: Downloading every symbol takes time; you can use --codes to limit
execution during testing.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from pytdx.hq import TdxHq_API
from pytdx.params import TDXParams
from pytdx.util.best_ip import select_best_ip


def select_market(code: str) -> int:
    """Infer market from ticker code (supports SH/SZ/BJ)."""
    code = str(code)
    if code.startswith(("4", "8")):
        return TDXParams.MARKET_BJ
    if code[0] in {"5", "6", "9"} or code.startswith(("009", "126", "110", "201", "202", "203", "204")):
        return TDXParams.MARKET_SH
    return TDXParams.MARKET_SZ


def fetch_all_codes(api: TdxHq_API, markets: Iterable[int]) -> List[str]:
    codes: List[str] = []
    for market in markets:
        total = api.get_security_count(market)
        start = 0
        while start < total:
            chunk = api.get_security_list(market, start)
            if not chunk:
                break
            codes.extend(row["code"] for row in api.to_df(chunk)["code"].tolist())
            start += len(chunk)
    return codes


def fetch_daily_bars(api: TdxHq_API, market: int, code: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    start = 0
    while True:
        bars = api.get_security_bars(9, market, code, start, 800)
        if not bars:
            break
        frames.append(api.to_df(bars))
        if len(bars) < 800:
            break
        start += len(bars)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["datetime"].str.slice(0, 10))
    df.insert(0, "code", code)
    df.insert(1, "market", market)
    return df


def fetch_xdxr(api: TdxHq_API, market: int, code: str) -> pd.DataFrame:
    data = api.get_xdxr_info(market, code)
    if not data:
        return pd.DataFrame()
    df = api.to_df(data)
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    return df


def fetch_finance(api: TdxHq_API, market: int, code: str) -> Dict[str, float]:
    info = api.get_finance_info(market, code)
    if not info:
        return {"liutongguben": 0.0}
    df = api.to_df([info])
    return df.iloc[0].to_dict()


def apply_forward_adjustment(bars: pd.DataFrame, xdxr: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return bars

    bars = bars.sort_values("date").reset_index(drop=True)
    factors = [1.0] * len(bars)

    # Only除权除息 (category 1) affects cash/stock adjustment for qianfuquan.
    if not xdxr.empty:
        actions = xdxr[xdxr["category"] == 1].copy()
        actions["date"] = pd.to_datetime(actions["date"])
        action_map = {d: row for d, row in actions.groupby("date")}
    else:
        action_map = {}

    cumulative = 1.0
    for idx in range(len(bars) - 1, -1, -1):
        row_date = bars.at[idx, "date"]
        factors[idx] = cumulative

        if row_date in action_map:
            action = action_map[row_date].iloc[0]
            close_px = float(bars.at[idx, "close"])
            fenhong = float(action.get("fenhong", 0) or 0)
            peigujia = float(action.get("peigujia", 0) or 0)
            songzhuangu = float(action.get("songzhuangu", 0) or 0)
            peigu = float(action.get("peigu", 0) or 0)

            cash_component = close_px - fenhong / 10.0
            # Protect against divide-by-zero while allowing negative adjustments.
            cash_component = cash_component if abs(cash_component) > 1e-8 else 1e-8

            share_increase = 1.0 + songzhuangu / 10.0 + peigu / 10.0
            cost_increase = close_px + peigujia * peigu / 10.0

            adj_ratio = (cost_increase / cash_component) / share_increase
            cumulative *= adj_ratio

    bars["qfq_factor"] = factors
    for col in ["open", "high", "low", "close"]:
        bars[f"{col}_qfq"] = bars[col] * bars["qfq_factor"]
    return bars


def attach_turnover_rate(bars: pd.DataFrame, finance_info: Dict[str, float]) -> pd.DataFrame:
    lt_shares = float(finance_info.get("liutongguben", 0) or 0)
    if lt_shares <= 0:
        bars["turnover_rate"] = None
        return bars
    # volume is reported in hands; multiply by 100 to convert to shares.
    bars["turnover_rate"] = bars["vol"] * 100 / lt_shares
    return bars


def save_csv(out_dir: Path, bars: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / f"{bars.iloc[0]['code']}.csv"
    bars.to_csv(outfile, index=False)


def download_all(out_dir: Path, codes: Iterable[str] | None = None) -> None:
    with TdxHq_API() as api:
        best_ip = select_best_ip()
        api.connect(best_ip["ip"], best_ip["port"])

        markets = [TDXParams.MARKET_SZ, TDXParams.MARKET_SH, TDXParams.MARKET_BJ]
        all_codes = list(codes) if codes else fetch_all_codes(api, markets)

        for code in all_codes:
            market = select_market(code)
            bars = fetch_daily_bars(api, market, code)
            if bars.empty:
                continue

            xdxr = fetch_xdxr(api, market, code)
            bars = apply_forward_adjustment(bars, xdxr)

            finance_info = fetch_finance(api, market, code)
            bars = attach_turnover_rate(bars, finance_info)

            save_csv(out_dir, bars)
            print(f"saved {code} ({len(bars)} rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download qianfuquan daily data for all markets.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for per-code CSV files")
    parser.add_argument(
        "--codes",
        nargs="*",
        help="Optional list of codes to limit download (e.g. 000001 600000 430047)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_all(args.out, args.codes)
