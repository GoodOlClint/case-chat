"""Magic-link auth for the web UI ([ADR 0005]).

A token is a signed `payload.signature` string (HMAC-SHA256 over a base64url
JSON payload `{sub, iat, exp}`). The operator issues a per-user link; first
visit verifies it and sets an httpOnly session cookie holding the same token.
Tokens are revocable per subject via a small JSON file.

CLI:
  python -m case_chat.web.auth issue --subject "Friend in TN" [--expires-days 14]
  python -m case_chat.web.auth revoke --subject "Friend in TN"
  python -m case_chat.web.auth list
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import time
from pathlib import Path
from typing import Any

from case_chat.config import settings


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(payload_b64: str) -> str:
    mac = hmac.new(settings.web_auth_secret.encode(), payload_b64.encode(), hashlib.sha256)
    return _b64e(mac.digest())


def issue_token(subject: str, *, expires_days: int | None = None) -> str:
    now = int(time.time())
    ttl_days = expires_days if expires_days is not None else settings.web_session_ttl_days
    payload = {"sub": subject, "iat": now, "exp": now + ttl_days * 86400}
    payload_b64 = _b64e(json.dumps(payload, separators=(",", ":")).encode())
    return f"{payload_b64}.{_sign(payload_b64)}"


def verify_token(token: str | None) -> dict[str, Any] | None:
    """Return the payload if the token is valid, unexpired, and not revoked."""
    if not token or "." not in token:
        return None
    payload_b64, sig = token.rsplit(".", 1)
    if not hmac.compare_digest(sig, _sign(payload_b64)):
        return None
    try:
        payload = json.loads(_b64d(payload_b64))
    except (ValueError, json.JSONDecodeError):
        return None
    if payload.get("exp", 0) < int(time.time()):
        return None
    if is_revoked(payload.get("sub", "")):
        return None
    return payload


# -- revocation -------------------------------------------------------------
def _revoked_path() -> Path:
    return Path(settings.web_revoked_path)


def _load_revoked() -> set[str]:
    p = _revoked_path()
    if not p.exists():
        return set()
    try:
        return set(json.loads(p.read_text()))
    except (ValueError, json.JSONDecodeError):
        return set()


def is_revoked(subject: str) -> bool:
    return subject in _load_revoked()


def revoke(subject: str) -> None:
    revoked = _load_revoked()
    revoked.add(subject)
    p = _revoked_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sorted(revoked), indent=2))


def make_link(subject: str, *, expires_days: int | None = None) -> str:
    token = issue_token(subject, expires_days=expires_days)
    return f"{settings.web_public_base_url.rstrip('/')}/auth?token={token}"


def main() -> None:
    ap = argparse.ArgumentParser(description="case-chat magic-link auth")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_issue = sub.add_parser("issue")
    p_issue.add_argument("--subject", required=True)
    p_issue.add_argument("--expires-days", type=int, default=None)
    p_rev = sub.add_parser("revoke")
    p_rev.add_argument("--subject", required=True)
    sub.add_parser("list")
    args = ap.parse_args()

    if args.cmd == "issue":
        if settings.web_auth_secret == "dev-insecure-change-me":
            print("WARNING: using the default dev secret — set CASECHAT_WEB_AUTH_SECRET in prod.\n")
        print(make_link(args.subject, expires_days=args.expires_days))
    elif args.cmd == "revoke":
        revoke(args.subject)
        print(f"revoked: {args.subject}")
    elif args.cmd == "list":
        print("revoked subjects:", sorted(_load_revoked()) or "(none)")


if __name__ == "__main__":
    main()
