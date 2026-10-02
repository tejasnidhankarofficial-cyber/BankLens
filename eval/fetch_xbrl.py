"""Ground-truth numbers from SEC companyfacts -> eval/xbrl_truth.json.
Requires env SEC_USER_AGENT="Name email" (SEC fair-access policy)."""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

CIKS = {"JPM": "0000019617", "BAC": "0000070858", "C": "0000831001", "WFC": "0000072971"}
TAGS = [
    "NetIncomeLoss", "Assets", "Revenues", "RevenuesNetOfInterestExpense", "InterestIncomeExpenseNet",
    "ProvisionForLoanLeaseAndOtherLosses", "StockholdersEquity",
]  # fmt: skip
YEARS = (2024, 2025)
OUT = Path(__file__).parent / "xbrl_truth.json"


def pick(entries: list[dict], year: int) -> dict | None:
    """Latest-filed 10-K FY value whose period ends in ``year`` (duration tags must span ~1 year)."""
    best = None
    for e in entries:
        if e.get("form") != "10-K" or e.get("fp") != "FY" or not e["end"].startswith(str(year)):
            continue
        if "start" in e:
            days = (date.fromisoformat(e["end"]) - date.fromisoformat(e["start"])).days
            if not 350 <= days <= 380:
                continue
        if best is None or e["filed"] > best["filed"]:
            best = e
    return best


def main() -> None:
    load_dotenv()
    ua = os.getenv("SEC_USER_AGENT")
    if not ua:
        sys.exit('Set SEC_USER_AGENT="Your Name your.email@example.com" in .env')
    truth: dict = {}
    for tk, cik in CIKS.items():
        r = requests.get(
            f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
            headers={"User-Agent": ua},
            timeout=60,
        )
        r.raise_for_status()
        gaap = r.json().get("facts", {}).get("us-gaap", {})
        truth[tk] = {}
        for y in YEARS:
            truth[tk][str(y)] = {}
            for tag in TAGS:
                e = pick(gaap.get(tag, {}).get("units", {}).get("USD", []), y)
                if e:
                    truth[tk][str(y)][tag] = {
                        "value_usd": e["val"],
                        "end": e["end"],
                        "filed": e["filed"],
                        "accn": e["accn"],
                    }
        print(tk, {y: len(v) for y, v in truth[tk].items()}, "tags found")
    OUT.write_text(json.dumps(truth, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
