import Image from '@theme/IdealImage';

# UI - Router Settings for Keys and Teams

Configure router settings at the key and team level to achieve granular control over routing behavior, fallbacks, retries, and other router configurations. This enables you to customize routing behavior for specific keys or teams without affecting global settings.

## Overview

Router Settings for Keys and Teams allows you to configure router behavior at different levels of granularity. Previously, router settings could only be configured globally, applying the same routing strategy, fallbacks, timeouts, and retry policies to all requests across your entire proxy instance.

With key-level and team-level router settings, you can now:

- **Customize routing strategies** per key or team (e.g., use `least-busy` for high-priority keys, `latency-based-routing` for others)
- **Configure different fallback chains** for different keys or teams
- **Set key-specific or team-specific timeouts** and retry policies
- **Apply different reliability settings** (cooldowns, allowed failures) per key or team
- **Override global settings** when needed for specific use cases

<Image img={require('../../img/ui_granular_router_settings.png')} />

## Summary

Router settings follow a **hierarchical resolution order**: **Keys > Teams > Global**. When a request is made:

1. **Key-level settings** are checked first. If router settings are configured for the API key being used, those settings are applied.
2. **Team-level settings** are checked next. If the key belongs to a team and that team has router settings configured, those settings are used (unless key-level settings exist).
3. **Global settings** are used as the final fallback. If neither key nor team settings are found, the global router settings from your proxy configuration are applied.

This hierarchical approach ensures that the most specific settings take precedence, allowing you to fine-tune routing behavior for individual keys or teams while maintaining sensible defaults at the global level.

## How Router Settings Resolution Works

Router settings are resolved in the following priority order:

### Resolution Order: Key > Team > Global

1. **Key-level router settings** (highest priority)
   - Applied when router settings are configured directly on an API key
   - Takes precedence over all other settings
   - Useful for individual key customization

2. **Team-level router settings** (medium priority)
   - Applied when the API key belongs to a team with router settings configured
   - Only used if no key-level settings exist
   - Useful for applying consistent settings across multiple keys in a team

3. **Global router settings** (lowest priority)
   - Applied from your proxy configuration file or database
   - Used as the default when no key or team settings are found
   - Previously, this was the only option available

## How to Configure Router Settings

### Configuring Router Settings for Keys

Follow these steps to configure router settings for an API key:

1. Navigate to [http://localhost:4000/ui/?login=success](http://localhost:4000/ui/?login=success)

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-24/61889da3-32de-4ebf-9cf3-7dc1db2fc993/ascreenshot_2492cf6d916a4ab98197cc8336e3a371_text_export.jpeg)

2. Click "+ Create New Key" (or edit an existing key)

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-24/61889da3-32de-4ebf-9cf3-7dc1db2fc993/ascreenshot_5a25380cf5044b4f93c146139d84403a_text_export.jpeg)

3. Click "Optional Settings"

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-24/e5eb5858-1cc1-4273-90bd-19ad139feebd/ascreenshot_33888989cfb9445bb83660f702ba32e0_text_export.jpeg)

4. Click "Router Settings"

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-24/d9eeca83-1f76-4fcf-bf61-d89edf3454d3/ascreenshot_825c7993f4b24949aee9b31d4a788d8a_text_export.jpeg)

5. Configure your desired router settings. For example, click "Fallbacks" to configure fallback models:

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-24/30ff647f-0254-4410-8311-660eef7ec0c4/ascreenshot_16966c8a0160473eb03e0f2c3b5c3afa_text_export.jpeg)

6. Click "Select a model to begin configuring fallbacks" and configure your fallback chain:

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-24/918f1b5b-c656-4864-98bd-d8c58924b6d9/ascreenshot_79ca6cd93be04033929f080e0c8d040a_text_export.jpeg)

### Configuring Router Settings for Teams

Follow these steps to configure router settings for a team:

1. Navigate to [http://localhost:4000/ui/?login=success](http://localhost:4000/ui/?login=success)

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-24/60a33a8c-2e48-4788-a1a2-e5bcffa98cca/ascreenshot_9e255ba48f914c72ae57db7d3c1c7cd5_text_export.jpeg)

2. Click "Teams"

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-24/60a33a8c-2e48-4788-a1a2-e5bcffa98cca/ascreenshot_070934fa9c17453987f21f58117e673b_text_export.jpeg)

3. Click "+ Create New Team" (or edit an existing team)

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-24/6f964ce2-f458-4719-a070-1af444ad92f5/ascreenshot_10f427f3106a4032a65d1046668880bd_text_export.jpeg)

4. Click "Router Settings"

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-24/a923c4ae-29f2-42b5-93ae-12f62d442691/ascreenshot_144520f2dd2f419dad79dffb1579ec04_text_export.jpeg)

5. Configure your desired router settings. For example, click "Fallbacks" to configure fallback models:

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-24/b062ecfa-bf5b-4c99-93a1-84b8b56fdb4c/ascreenshot_ea9acbc4e75448709b64a22addfb4157_text_export.jpeg)

6. Click "Select a model to begin configuring fallbacks" and configure your fallback chain:

![](https://colony-recorder.s3.amazonaws.com/files/2026-01-24/67ca2655-4e82-4f93-be9a-7244ad22640f/ascreenshot_4fdbed826cd546d784e8738626be835d_text_export.jpeg)

## Use Cases

### Different Routing Strategies per Key

Configure different routing strategies for different use cases:

- **High-priority production keys**: Use `latency-based-routing` for optimal performance
- **Development keys**: Use `simple-shuffle` for simplicity
- **Cost-sensitive keys**: Use `cost-based-routing` to minimize expenses

### Team-Level Consistency

Apply consistent router settings across all keys in a team:

- Set team-wide fallback chains for reliability
- Configure team-specific timeout policies
- Apply uniform retry policies across team members

### Override Global Settings

Override global settings for specific scenarios:

- Production keys may need stricter timeout policies than development
- Certain teams may require different fallback models
- Individual keys may need custom retry policies for specific use cases

### Gradual Rollout

Test new router settings on specific keys or teams before applying globally:

- Configure new routing strategies on a test key first
- Validate fallback chains on a small team before global rollout
- A/B test different timeout values across different keys

## Related Features

- [Router Settings Reference](./config_settings.md#router_settings---reference) - Complete reference of all router settings
- [Load Balancing](./load_balancing.md) - Learn about routing strategies and load balancing
- [Reliability](./reliability.md) - Configure fallbacks, retries, and error handling
- [Keys](./keys.md) - Manage API keys and their settings
- [Teams](./teams.md) - Organize keys into teams
