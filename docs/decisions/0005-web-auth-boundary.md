# 0005 — Web UI auth boundary (external sharing)

Status: Accepted · 2026-06-13

## Context
The web UI will be shared beyond localhost — the operator wants to send a link
to a lawyer reviewer in Tennessee. That makes the web app externally reachable,
so it needs an authentication boundary, and every other service must stay
internal.

Sovereignty is satisfied: the POC holds only fictional synthetic data +
non-sensitive domain-knowledge *reference* text — no real `case-data/` — so an
external reviewer is within bounds.

## Decision
- **Auth model: per-user magic-link tokens.** The operator issues each reviewer
  a unique **signed** access link (HMAC over `{subject, issued_at, optional
  expiry}`). First visit verifies the token, sets an **httpOnly, Secure,
  SameSite=Lax session cookie**, and redirects to scrub the token from the URL
  (so it doesn't linger in history/referrer). Tokens are **revocable per user**
  (a server-side revocation list / token version) without affecting others.
- **Exposure: Cloudflare Tunnel** from the 5090 box → HTTPS URL, **no inbound
  ports opened**. Only the web port is published.
- **Everything else stays bound to localhost on the box:** vLLM `/v1`, Qdrant,
  the stdio MCP server, the SQLite dataset. The web app is the *only* process
  reachable from the tunnel, and only after auth.

## Consequences
- A small token CLI: `issue-link --subject "<name>" [--expires <days>]` prints a
  shareable URL; `revoke --subject` invalidates it.
- Secrets in config: `web_auth_secret` (HMAC key), `web_session_ttl`,
  `web_public_base_url` (for link generation). No secret in the repo.
- Middleware gates every route except the token-exchange endpoint and static
  assets; unauthenticated requests get a minimal "ask the owner for a link"
  page, not the app.
- HTTPS is required (cookie `Secure`); the Cloudflare Tunnel terminates TLS.

## Alternatives rejected
- **Single shared password** — simplest, but not per-user; can't revoke one
  reviewer without rotating for everyone.
- **Google OAuth + email allowlist** — no passwords, but heaviest setup and
  forces every reviewer to have/use a Google account.
