---
slug: host-header-auth-bypass
title: "Fixed in 1.84.0+ - Version Update: Authentication Bypass via Host Header Injection (GHSA-4xpc-pv4p-pm3w)"
date: 2026-06-01T12:00:00
authors:
  - krrish
  - ishaan-alt
  - yuneng
description: "Disclosure of a Host-header authentication bypass in the LiteLLM proxy. Addressed in v1.84.0. Very limited deployments are potentially affected, and no LiteLLM Cloud customers were affected."
tags: [security]
hide_table_of_contents: false
---

The update addressing this Host-header authentication bypass in the LiteLLM proxy shipped in `v1.84.0`, with follow-up path-handling hardening completed and backported across the maintained release lines in `v1.84.3`, `v1.85.2`, `v1.86.2`, and `v1.83.10-stable.patch.3`. The potential for bypass was limited to deployments with the three specific conditions below. The bypass was reported by Le The Thang (KCSC) and Kim Ngoc Chung (One Mount Group).

The conditions could allow unauthenticated access to protected management routes when the proxy listener was reachable with an arbitrary `Host` header.

No LiteLLM Cloud customers were affected. The update was deployed across all LiteLLM Cloud environments - backported to the release lines in use - ahead of this publication.

* Addressed in: `v1.84.0`
* Recommended: the latest release; follow-up path-handling hardening was backported in `v1.84.3`, `v1.85.2`, and `v1.86.2`
* Action: upgrade to `v1.84.0` or later. No configuration change is required.

More info on the advisory is here: https://github.com/BerriAI/litellm/security/advisories/GHSA-4xpc-pv4p-pm3w. CVE: https://www.cve.org/CVERecord?id=CVE-2026-48710.

{/* truncate */}

## TL;DR

* A crafted `Host` header could make the proxy's auth gate evaluate a different route from the one it served, allowing potential unauthenticated access to protected management routes.
* The update shipped in `v1.84.0`. Follow-up path-handling hardening was backported in `v1.84.3`, `v1.85.2`, and `v1.86.2`; upgrading to the latest release is recommended.
* Potential bypass requires reaching the proxy listener with an arbitrary `Host` header. Fronting the proxy with infrastructure that validates or normalizes `Host` reduces potential for bypass depending on configuration, but is not a comprehensive substitute for upgrading.
* No LiteLLM Cloud customers were affected.

## Summary

The proxy's auth layer derived the effective route from `request.url.path` in `litellm/proxy/auth/auth_utils.py::get_request_route()`, which Starlette reconstructs from the `Host` header. A crafted `Host` header could therefore make the auth gate evaluate a different route from the one FastAPI actually dispatched, causing a protected management route to be treated as public.

Potential bypass requires an actor to reach the proxy listener with an arbitrary `Host` header. Fronting the proxy with infrastructure that validates or normalizes the `Host` header reduces potential for bypass, though whether it fully blocks the bypass depends on the specific configuration. The LiteLLM Python SDK is not affected; only the proxy server is in limited scope.

## Additional hardening

The primary update in `v1.84.0` addressed the reported potential for bypass by deriving the request route from the ASGI scope path rather than the `Host`-reconstructed URL. As additional follow-up, we audited every other location in the proxy that derived a route from the request URL and moved them onto the same hardened resolution. This closes the long tail of the potential for bypass and was backported across the maintained release lines in `v1.84.3`, `v1.85.2`, `v1.86.2`, and `v1.83.10-stable.patch.3`. We recommend upgrading to one of these releases for comprehensive mitigation.

## Am I affected?

You are potentially affected only if **all** of the following are true:

- You run the **LiteLLM proxy server** (not just the Python SDK).
- You are on a version **earlier than `v1.84.0`**.
- The proxy listener is reachable by untrusted clients.

You are **not** remotely open to potential bypass if the proxy listener is not reachable by untrusted clients — for example, it is bound to a private network or sits behind a gateway that requires its own authentication.

Fronting the proxy with infrastructure that validates or normalizes the `Host` header (a CDN/WAF, a reverse proxy with `server_name` allowlists, or a host-based load balancer) reduces potential for bypass, but whether it fully mitigates against potential bypass depends on the configuration.

## What to do

1. Upgrade to `v1.84.0` or later. Upgrading to the latest release is recommended, which includes the follow-up hardening backported in `v1.84.3`, `v1.85.2`, and `v1.86.2`.
2. If your proxy was reachable from an untrusted network on an affected version, rotate any API keys created during the exposure window and review your management audit logs for unexpected key, user, or settings changes.

## Mitigations

If you cannot upgrade immediately, to better mitigate the potential for bypass, we recommend placing the proxy behind an upstream component that validates or normalizes the `Host` header before forwarding:

- a CDN or WAF (e.g. Cloudflare),
- a reverse proxy with explicit `server_name` allowlists (nginx, Caddy, Traefik),
- a cloud load balancer with host-based routing rules,

or otherwise restrict network access to the proxy listener. Note this is a per-deployment property: a reverse proxy that forwards the client `Host` unchanged (e.g. nginx `proxy_set_header Host $host;`) may not comprehensively protect your use from this potential. Treat upgrading as the elimination of any potential for bypass and edge filtering only as a stopgap.
