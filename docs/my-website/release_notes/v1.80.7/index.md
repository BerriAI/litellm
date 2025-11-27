---
title: "[PREVIEW] v1.80.7.rc.1 - New RAG API"
slug: "v1-80-7"
date: 2025-11-27T10:00:00
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

import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

## Deploy this version

<Tabs>
<TabItem value="docker" label="Docker">

```showLineNumbers title="docker run litellm"
docker run \
-e STORE_MODEL_IN_DB=True \
-p 4000:4000 \
ghcr.io/berriai/litellm:v1.80.7.rc.1
```

</TabItem>

<TabItem value="pip" label="Pip">

```showLineNumbers title="pip install litellm"
pip install litellm==1.80.7
```

</TabItem>
</Tabs>

---


### Organization Usage

<Image
img={require('../../img/release_notes/organization_usage.png')}
style={{width: '100%', display: 'block', margin: '2rem auto'}}
/>

Users can now filter usage statistics by organization, providing the same granular filtering capabilities available for teams.

**Details:**

- Filter usage analytics, spend logs, and activity metrics by organization ID
- View organization-level breakdowns alongside existing team and user-level filters
- Consistent filtering experience across all usage and analytics views

---
