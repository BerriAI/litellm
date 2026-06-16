---
slug: stability
title: "June Stability Update:We're Making Stability a First-Class Citizen at
LiteLLM"
date: 2026-06-15T10:00:00
authors:
  - ishaan-alt
  - varoon
description: ""
tags: []
hide_table_of_contents: false
---

Over the past few months, we've heard our users report more bugs and regressions. We take that feedback seriously, and today we're sharing exactly what we're doing about it.

We're kicking off a stability sprint for LiteLLM with one bar in mind: 0 reported regressions by our next release on August 29th. The sprint has 2 goals:

- Close 20 reported bugs in core functionality - [here](https://github.com/BerriAI/litellm/issues/30484)
- Address the root cause of underlying bugs in 3 core components - MCP, Gateway, and UI

## What class of bugs are we driving down?

Over this sprint we're driving down 3 classes of bugs:

- **MCP Authentication:** View/List Tools did not consistently work across all our supported MCP auth methods.
- **Gateway Authentication:** Team IDs are not reliably on every request trace. As a result, some requests and budgets are not accurately tracked to a team.
- **UI Forms:** Today when users hit save on a form, it can accidentally wipe out other fields on the form, across keys, teams, and users.

## MCP Authentication: Consistent behavior across all MCP Authentication Methods

Solution: We've identified that the root cause of bugs across MCPs is that we maintain 5 different code paths, one per authentication method. To fix this and restore connection reliability, we're refactoring this into one code path that resolves MCP credentials across all supported authentication methods. The result: tools list and call reliably, no matter which auth method you use.

## AI Gateway Authentication: Spend is always attributed to the right team

Solution: We identified that the authentication layer makes 5+ DB lookups to resolve the exact key, user, team, and team member making a request. To fix this, we're resolving caller identity once, into a single record that every check and log reads from. This cuts identity lookups roughly in half, and means spend is always attributed to the team that made the request.

## UI: Edits change only what you touched

Solution: One of the root causes of UI bugs on form save is that our data shapes across the frontend and backend are not consistent. To fix this, we're refactoring so frontend and backend types are 100% in sync and read from the same source of truth. The result: a save changes only the field you edited, nothing else.

## How you'll know it worked

We'll report back at the August 29th release on exactly where each of these stands. You shouldn't have to take our word for it.

## Why now

We've grown fast. And fast growth in a complex system means bugs accumulate if you're not deliberate about paying them down. This sprint is us being deliberate.

We're also being public about it because you deserve to know what's being fixed and when. Stability is infrastructure. We're treating it that way.

## Want us to fix something?

Every item above came from real user reports. If there's a bug affecting you that isn't on this list, comment on the [GitHub issue](https://github.com/BerriAI/litellm/issues/30484). We're actively triaging!