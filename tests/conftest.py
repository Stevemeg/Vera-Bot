import json
from pathlib import Path

import pytest

DATASET_DIR = Path(__file__).parent.parent.parent / "dataset"
if not DATASET_DIR.exists():
    # fallback if the submission dir is copied without the sibling dataset
    DATASET_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def dentists_category():
    return json.loads((DATASET_DIR / "categories" / "dentists.json").read_text())


@pytest.fixture(scope="session")
def all_categories():
    out = {}
    for f in (DATASET_DIR / "categories").glob("*.json"):
        d = json.loads(f.read_text())
        out[d["slug"]] = d
    return out


@pytest.fixture(scope="session")
def merchants():
    data = json.loads((DATASET_DIR / "merchants_seed.json").read_text())
    return {m["merchant_id"]: m for m in data["merchants"]}


@pytest.fixture(scope="session")
def customers():
    data = json.loads((DATASET_DIR / "customers_seed.json").read_text())
    return {c["customer_id"]: c for c in data["customers"]}


@pytest.fixture(scope="session")
def triggers():
    data = json.loads((DATASET_DIR / "triggers_seed.json").read_text())
    return {t["id"]: t for t in data["triggers"]}


@pytest.fixture()
def drmeera(merchants):
    return merchants["m_001_drmeera_dentist_delhi"]


@pytest.fixture()
def priya(customers):
    return customers["c_001_priya_for_m001"]
