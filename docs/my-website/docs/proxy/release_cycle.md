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

### What is considered a 'minor' bump vs. 'patch' bump?

- 'patch' bumps: extremely minor addition that doesn't affect any existing functionality or add any user-facing features. (e.g. a 'created_at' column in a database table)
- 'minor' bumps: add a new feature or a new database table that is backward compatible.
- 'major' bumps: break backward compatibility.

### Enterprise Support


- Stable releases come out every week. Once a new one is available, we no longer provide support for an older one. 
- If there is a MAJOR change (according to semvar conventions - e.g. 1.x.x -> 2.x.x), we can provide support for upto 90 days on the prior stable image. 
