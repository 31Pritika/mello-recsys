"""
Smoke test for the running API.

Picks a real user id from cluster_labels artifact, hits every endpoint,
and prints the responses. Exits 0 on success, 1 on any failure.

Usage (with uvicorn already running on port 8000):
    python smoke_test.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import pickle
import numpy as np

BASE = "http://127.0.0.1:8000"


def get(path: str) -> dict:
    url = BASE + path
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def main() -> None:
    # Pick two real user ids from saved artifacts
    with open("model_artifacts/user_to_idx.joblib", "rb") as f:
        user_to_idx: dict = pickle.load(f)
    users = sorted(user_to_idx.keys())
    user_a, user_b = users[0], users[1]

    errors = 0

    print("=" * 50)
    print("SMOKE TEST")
    print("=" * 50)

    # GET /
    print(f"\nGET /")
    try:
        r = get("/")
        print(json.dumps(r, indent=2))
        assert r["status"] == "ok", "status != ok"
        assert r["n_users"] > 0
        assert r["n_movies"] > 0
    except Exception as e:
        print(f"  FAIL: {e}")
        errors += 1

    # GET /recommendations/{user_id}
    print(f"\nGET /recommendations/{user_a}?k=5")
    try:
        r = get(f"/recommendations/{user_a}?k=5")
        print(json.dumps(r, indent=2))
        assert len(r["recommendations"]) == 5
    except Exception as e:
        print(f"  FAIL: {e}")
        errors += 1

    # GET /users/{user_id}/compatibility/{other}
    print(f"\nGET /users/{user_a}/compatibility/{user_b}")
    try:
        r = get(f"/users/{user_a}/compatibility/{user_b}")
        print(json.dumps(r, indent=2))
        assert -1.0 <= r["cosine_similarity"] <= 1.0
    except Exception as e:
        print(f"  FAIL: {e}")
        errors += 1

    # GET /users/{user_id}/cluster
    print(f"\nGET /users/{user_a}/cluster")
    try:
        r = get(f"/users/{user_a}/cluster")
        print(json.dumps(r, indent=2))
        assert "cluster_id" in r
        assert r["cluster_size"] > 0
    except Exception as e:
        print(f"  FAIL: {e}")
        errors += 1

    # GET unknown user → expect 404
    print(f"\nGET /recommendations/999999999 (unknown user, expect 404)")
    try:
        get("/recommendations/999999999")
        print("  FAIL: expected 404 but got 200")
        errors += 1
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"  OK: got 404 as expected")
        else:
            print(f"  FAIL: expected 404, got {e.code}")
            errors += 1

    print("\n" + "=" * 50)
    if errors == 0:
        print("All smoke tests PASSED ✓")
        sys.exit(0)
    else:
        print(f"{errors} smoke test(s) FAILED ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
