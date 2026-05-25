# Helm Chart for LiteLLM (experimental, microservices split)

> [!WARNING]
> **This chart is experimental.** It splits a LiteLLM deployment into multiple microservices (proxy, UI, etc.) and is maintained by the LiteLLM team. The associated container images are published as **dev-only releases** (see the [BerriAI packages page](https://github.com/orgs/BerriAI/packages?repo_name=litellm) for available `litellm/ch*` images) and are **not** the same as the stable `docker.litellm.ai/berriai/litellm` image used by the monolithic chart.
>
> For production use, prefer the stable monolithic chart at [`deploy/charts/litellm-helm/`](../../deploy/charts/litellm-helm/) until this chart's images move out of dev-only status.

## Which chart should I use?

| Chart | Status | Maintainer | Use when |
|---|---|---|---|
| [`deploy/charts/litellm-helm/`](../../deploy/charts/litellm-helm/) | **Stable** | Community + LiteLLM team | Default for most users; production-ready single-Deployment. |
| `helm/litellm/` (this chart) | **Experimental** | LiteLLM team | You need a microservices split deployment and are comfortable running dev-only container images. |

See the [microservices helm guide](https://docs.litellm.ai/docs/proxy/microservices_helm) for deployment instructions and the planned architecture.

## Background

This README was added in response to [#28619](https://github.com/BerriAI/litellm/issues/28619), which flagged that two helm charts existed in the repository with no documentation explaining when to use which one. The clarification here mirrors the maintainer's comment on that issue.

## Status / stability tracking

When this chart graduates from experimental and its container images receive stable releases, this README should be updated to reflect the new status and the canonical chart recommendation may shift.
