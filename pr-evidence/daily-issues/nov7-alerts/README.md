# Alerts walkthrough

[Watch the 173-second walkthrough](alerts-demo.mp4). Captions cover all five alert families, grouped investigations, Activity, monitoring settings, model migrations, and the notification menu. The original resource icon grid is restored and stays pinned while the menu scrolls.

The browser uses real HTTP responses from two proxy workers and an isolated PostgreSQL copy. Historical traffic is synthetic. No inference calls were made during this recording. This demonstrates the product flow, not production throughput or memory safety.

The frontend is `a5826182f6d59e6d3d28bb8bb9b72620d6c01368`. The backend is `f1cd795a923f0d203e9d02fb4bd08144af537109`. The before screenshot uses frontend `d44a94ee0068f90c087e79879c4a7972b38aef25` against the same backend and data. Later main-branch integration and backend fixes are covered by separate checks.

## What is shown

Open Alerts, then select the key spend alert. The established key spent $100 against a $2 daily baseline, a 50-fold increase. Its related request increase appears in the same investigation. Activity uses the saved key and date range.

Return to Alerts and open provider latency. Two models share an independently confirmed provider-level increase. Open error rate and reduced traffic examples to inspect their saved comparisons. Open retirement, then Migration details, to reach the existing model-deprecation workflow.

Settings shows four performance families with defaults. Editing Unexpected spend exposes its threshold. Email delivery settings opens the existing delivery configuration. No threshold is saved during the walkthrough.

The notification menu contains Alerts and collapsed product updates. Docs, Releases, Blog, Slack, Discord, GitHub, and Deployment health remain visible below the scrollable content. The last four scenes open key, team, user, and model Activity directly.

## Verification

All five families and all four entity Activity views rendered. Each Activity request total matched its API response. The 50-fold spend example has a positive established baseline. No browser exception occurred. Desktop and 390-pixel layouts were checked in light and dark themes, without document overflow. The footer's vertical position remained unchanged while notification content scrolled.

[Verification record](verification.json) contains source identities and observed totals. [Chapters](chapters.json) contains timestamps and captions. The recording is silent with captions.
