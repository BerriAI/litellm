---
title: v1.65.0 - Team Model Add - update
slug: v1.65.0
date: 2025-03-28T10:00:00
authors:
  - name: Krrish Dholakia
    title: CEO, LiteLLM
    url: https://www.linkedin.com/in/krish-d/
    image_url: https://media.licdn.com/dms/image/v2/D4D03AQGrlsJ3aqpHmQ/profile-displayphoto-shrink_400_400/B4DZSAzgP7HYAg-/0/1737327772964?e=1749686400&v=beta&t=Hkl3U8Ps0VtvNxX0BNNq24b4dtX5wQaPFp6oiKCIHD8
  - name: Ishaan Jaffer
    title: CTO, LiteLLM
    url: https://www.linkedin.com/in/reffajnaahsi/
    image_url: https://pbs.twimg.com/profile_images/1613813310264340481/lz54oEiB_400x400.jpg
tags: [management endpoints, team models, ui]
hide_table_of_contents: false
---

import Image from '@theme/IdealImage';

v1.65.0 updates the `/model/new` endpoint to prevent non-team admins from creating team models.

This means that only proxy admins or team admins can create team models.

## Additional Changes

- Allows team admins to call `/model/update` to update team models.
- Allows team admins to call `/model/delete` to delete team models.
- Introduces new `user_models_only` param to `/v2/model/info` - only return models added by this user.


These changes enable team admins to add and manage models for their team on the LiteLLM UI + API.


<Image img={require('../../img/release_notes/team_model_add.png')} />