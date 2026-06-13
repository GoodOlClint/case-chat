"""Tests for magic-link token signing/verification + revocation."""

from __future__ import annotations

import json
import time

import pytest

from case_chat import config
from case_chat.web import auth


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(config.settings, "web_auth_secret", "test-secret-123")
    monkeypatch.setattr(config.settings, "web_revoked_path", str(tmp_path / "revoked.json"))
    yield


def test_roundtrip_valid() -> None:
    tok = auth.issue_token("Alice")
    payload = auth.verify_token(tok)
    assert payload and payload["sub"] == "Alice"


def test_tampered_signature_rejected() -> None:
    tok = auth.issue_token("Alice")
    body, _sig = tok.rsplit(".", 1)
    assert auth.verify_token(f"{body}.deadbeef") is None


def test_wrong_secret_rejected(monkeypatch) -> None:
    tok = auth.issue_token("Alice")
    monkeypatch.setattr(config.settings, "web_auth_secret", "different-secret")
    assert auth.verify_token(tok) is None


def test_expired_rejected() -> None:
    # Hand-build a token with a past expiry.
    past = {"sub": "Alice", "iat": int(time.time()) - 100, "exp": int(time.time()) - 10}
    body = auth._b64e(json.dumps(past).encode())
    tok = f"{body}.{auth._sign(body)}"
    assert auth.verify_token(tok) is None


def test_revoked_subject_rejected() -> None:
    tok = auth.issue_token("Mallory")
    assert auth.verify_token(tok) is not None
    auth.revoke("Mallory")
    assert auth.verify_token(tok) is None


def test_garbage_rejected() -> None:
    assert auth.verify_token(None) is None
    assert auth.verify_token("not-a-token") is None
    assert auth.verify_token("a.b.c") is None


def test_make_link_contains_token(monkeypatch) -> None:
    monkeypatch.setattr(config.settings, "web_public_base_url", "https://demo.example")
    link = auth.make_link("Bob")
    assert link.startswith("https://demo.example/auth?token=")
    token = link.split("token=", 1)[1]
    assert auth.verify_token(token)["sub"] == "Bob"
