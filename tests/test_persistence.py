import json
import os
import tempfile

from app.store import ContextStore, StaleVersionError


def test_persistence_disabled_by_default_no_env_var(monkeypatch):
    monkeypatch.delenv("VERA_PERSIST_PATH", raising=False)
    store = ContextStore()
    store.put("merchant", "m1", 1, {"a": 1})
    assert store._persist_path is None  # pure in-memory, matching challenge default


def test_context_survives_simulated_restart_when_persistence_enabled():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "snapshot.json")

        store1 = ContextStore(persist_path=path)
        store1.put("merchant", "m1", 1, {"name": "Test Clinic"})
        store1.put("category", "dentists", 2, {"slug": "dentists"})

        # Simulate a process restart: brand new ContextStore instance,
        # same disk path, as would happen if the container process crashed
        # and a supervisor restarted uvicorn.
        store2 = ContextStore(persist_path=path)
        assert store2.get("merchant", "m1") == {"name": "Test Clinic"}
        assert store2.get("category", "dentists") == {"slug": "dentists"}
        assert store2.get_version("category", "dentists") == 2


def test_snapshot_survives_multiple_updates_in_order():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "snapshot.json")
        store1 = ContextStore(persist_path=path)
        store1.put("merchant", "m1", 1, {"v": 1})
        store1.put("merchant", "m1", 2, {"v": 2})
        store1.put("merchant", "m1", 3, {"v": 3})

        store2 = ContextStore(persist_path=path)
        assert store2.get_version("merchant", "m1") == 3
        assert store2.get("merchant", "m1") == {"v": 3}


def test_teardown_deletes_the_snapshot_file_not_just_memory():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "snapshot.json")
        store = ContextStore(persist_path=path)
        store.put("merchant", "m1", 1, {"a": 1})
        assert os.path.exists(path)

        store.teardown()
        assert not os.path.exists(path)  # deleted, not just cleared in memory

        # A fresh instance after teardown must NOT rehydrate stale data —
        # confirms teardown genuinely removes persisted state, satisfying
        # "must not persist context after the test ends."
        store2 = ContextStore(persist_path=path)
        assert store2.get("merchant", "m1") is None


def test_corrupt_snapshot_file_does_not_crash_startup():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "snapshot.json")
        with open(path, "w") as f:
            f.write("{not valid json[[[")
        store = ContextStore(persist_path=path)  # must not raise
        assert store.get("merchant", "anything") is None
        # And it must still work normally afterward.
        store.put("merchant", "m1", 1, {"a": 1})
        assert store.get("merchant", "m1") == {"a": 1}


def test_stale_version_rejection_still_works_with_persistence_enabled():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "snapshot.json")
        store = ContextStore(persist_path=path)
        store.put("merchant", "m1", 3, {"v": 3})
        try:
            store.put("merchant", "m1", 2, {"v": 2})
            assert False, "should have raised"
        except StaleVersionError as e:
            assert e.current_version == 3
