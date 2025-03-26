# Release Cycle

Litellm Proxy has the following release cycle:

- `v1.x.x-nightly`: These are releases which pass ci/cd. 
- `v1.x.x.rc`: These are releases which pass ci/cd + [manual review](https://github.com/BerriAI/litellm/discussions/8495#discussioncomment-12180711).
- `v1.x.x:main-stable`: These are releases which pass ci/cd + manual review + 3 days of production testing.

In production, we recommend using the latest `v1.x.x:main-stable` release.


Follow our release notes [here](https://github.com/BerriAI/litellm/releases).


## FAQ

### Is there a release schedule for LiteLLM stable release?

Stable releases come out every week (typically Sunday)

