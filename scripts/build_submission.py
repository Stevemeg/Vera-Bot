"""
Build submission.jsonl from expanded/test_pairs.json using the same
in-process pipeline the HTTP API uses (opportunities.evaluate_trigger +
composer.compose), so the offline artifact and the live bot are
guaranteed to agree.

Usage:
    python scripts/build_submission.py --expanded-dir ../expanded --out submission.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.composer import compose, safe_fallback
from app.opportunities import evaluate_trigger
from app import validators


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--expanded-dir", default="../expanded")
    ap.add_argument("--out", default="submission.jsonl")
    ap.add_argument("--now", default="2026-04-26T10:00:00Z")
    args = ap.parse_args()

    base = Path(args.expanded_dir)
    categories = {load_json(f)["slug"]: load_json(f) for f in (base / "categories").glob("*.json")}
    merchants = {load_json(f)["merchant_id"]: load_json(f) for f in (base / "merchants").glob("*.json")}
    customers = {load_json(f)["customer_id"]: load_json(f) for f in (base / "customers").glob("*.json")}
    triggers = {load_json(f)["id"]: load_json(f) for f in (base / "triggers").glob("*.json")}

    pairs = load_json(base / "test_pairs.json")["pairs"]

    lines = []
    for pair in pairs:
        test_id = pair["test_id"]
        trigger = triggers.get(pair["trigger_id"])
        merchant = merchants.get(pair["merchant_id"])
        customer = customers.get(pair["customer_id"]) if pair.get("customer_id") else None
        category = categories.get(merchant.get("category_slug")) if merchant else None

        if trigger is None or merchant is None or category is None:
            lines.append(
                {
                    "test_id": test_id,
                    "body": "",
                    "cta": "none",
                    "send_as": "vera",
                    "suppression_key": "",
                    "rationale": "Skipped: required context missing from expanded dataset.",
                }
            )
            continue

        opp = evaluate_trigger(trigger, category, merchant, customer, args.now, suppressed=False)
        if not opp.eligible:
            lines.append(
                {
                    "test_id": test_id,
                    "body": "",
                    "cta": "none",
                    "send_as": "vera",
                    "suppression_key": trigger.get("suppression_key", ""),
                    "rationale": f"No message sent: ineligible ({opp.ineligible_reason}).",
                }
            )
            continue

        msg = compose(category, merchant, trigger, customer, opp)
        validation = validators.validate(msg.body, msg.cta, msg.rationale, category)
        if not validation.ok:
            msg = safe_fallback(category, merchant, trigger, opp)
        lines.append(
            {
                "test_id": test_id,
                "body": msg.body,
                "cta": msg.cta,
                "send_as": msg.send_as,
                "suppression_key": trigger.get("suppression_key", ""),
                "rationale": msg.rationale,
            }
        )

    with open(args.out, "w") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Wrote {len(lines)} lines to {args.out}")


if __name__ == "__main__":
    main()
