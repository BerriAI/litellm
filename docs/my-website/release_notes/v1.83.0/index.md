---
title: "v1.83.0 - Official Release (Post Supply Chain Incident)"
slug: "v1-83-0"
date: 2026-03-31T00:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://pbs.twimg.com/profile_images/1298587542745358340/DZv3Oj-h_400x400.jpg
  - name: Ishaan Jaff
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
hide_table_of_contents: false
---

## Deploy this version

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
<TabItem value="docker" label="Docker">

``` showLineNumbers title="docker run litellm"
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
ghcr.io/berriai/litellm:main-1.83.0-nightly
```

</TabItem>
<TabItem value="pip" label="Pip">

``` showLineNumbers title="pip install litellm"
pip install litellm==1.83.0
```

</TabItem>
</Tabs>

## Context: First Release After Supply Chain Incident

v1.83.0 is the first LiteLLM release built and published through our new [CI/CD v2 pipeline](https://docs.litellm.ai/blog/ci-cd-v2-improvements), following the [supply chain incident on March 24](https://docs.litellm.ai/blog/security-update-march-2026).

We paused all releases for one week while we:
1. Completed a forensic review with [Mandiant](https://www.mandiant.com/) and [Veria Labs](https://verialabs.com/)
2. Rebuilt the release pipeline from scratch with isolated environments and ephemeral credentials
3. Verified the codebase contains no indicators of compromise

If you have questions about this release or the incident, see our [Security Townhall post](https://docs.litellm.ai/blog/security-townhall-updates) or reach out at `security@berri.ai`.

---

## Links

- **PyPI**: [litellm 1.83.0](https://pypi.org/project/litellm/1.83.0/)
- **Security update**: [Supply chain incident report](https://docs.litellm.ai/blog/security-update-march-2026)
- **Security townhall**: [What happened, what we've done, what comes next](https://docs.litellm.ai/blog/security-townhall-updates)
- **CI/CD v2**: [Announcing CI/CD v2 for LiteLLM](https://docs.litellm.ai/blog/ci-cd-v2-improvements)
- **April stability sprint**: [Help us plan](https://github.com/BerriAI/litellm/issues/24825)

