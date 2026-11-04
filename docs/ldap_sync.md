# LDAP team synchronization

LDAP sync periodically reads a selected set of directory groups, creates missing LiteLLM teams, and reconciles their memberships using accounts that already exist in LiteLLM. Login continues to use your existing authentication setup.

Install the proxy dependencies with `pip install 'litellm[proxy]'`. A database and shared Redis are required, including when running multiple workers in one proxy. Each worker uses the existing job lock so only one worker synchronizes at a time.

```yaml
general_settings:
  coordination_redis:
    host: os.environ/REDIS_HOST
    port: os.environ/REDIS_PORT
  ldap_sync:
    enabled: true
    server_url: ldaps://ldap.example.org:636
    group_search_base: ou=groups,dc=example,dc=org
    group_filter: "(cn=project-*)"
    user_search_base: ou=people,dc=example,dc=org
    group_id_attribute: cn
    group_name_attribute: cn
    member_attribute: member
    user_match_attribute: mail
    litellm_match_field: user_email
    interval_minutes: 30
    dry_run: true

litellm_settings:
  default_team_params:
    models: [your-approved-model]
    max_budget: 100
    budget_duration: 30d
```

The example uses anonymous LDAP reads over TLS. For a service account, add `bind_dn: os.environ/LDAP_BIND_DN` and `bind_password: os.environ/LDAP_BIND_PASSWORD`. Credentials cannot be sent through an unencrypted LDAP connection. With `ldap://`, set `start_tls: true` for a service-account bind. LDAPS and StartTLS verify the server certificate, using the system trust store or the optional `ca_cert_path` CA bundle. Anonymous `ldap://` connections without StartTLS are also supported.

Start with `dry_run: true`, restart the proxy, and review the planned team creation and member additions/removals in its normal logs. Set `dry_run: false` and restart to apply changes. The first run starts shortly after startup and later runs follow `interval_minutes`. Logs also report matched and unmatched users and skipped groups. There is no separate sync UI or run-history table.

## Identity and membership

`group_id_attribute` is matched exactly to LiteLLM's `team_id`. The default `cn` must be a unique, stable string across your selected groups. A group named `project-alpha` therefore creates or updates the team with ID `project-alpha`. A manually created team with that same ID is included in the sync. Matching only the display name does not adopt a different team.

`group_name_attribute` supplies the new team's display name. Team creation inherits `default_team_params`, the same way SCIM-created teams do. Existing team names, model permissions, budgets and other settings remain unchanged.

By default, `member` contains full user DNs. The sync reads each referenced user entry and compares its `mail` attribute to existing LiteLLM `user_email` values, ignoring email case. `litellm_match_field` may instead be `user_id` or `sso_user_id`, matched exactly. Set `user_match_attribute` to the LDAP attribute carrying the corresponding identity. Missing LiteLLM accounts are skipped and counted in the logs, then picked up by a later run after normal onboarding. Duplicate matches skip the affected group rather than choosing an account.

For POSIX groups, set `member_attribute: memberUid`. These values are looked up under `user_search_base` using `member_lookup_attribute`, which defaults to `uid`. Other membership attributes such as `uniqueMember` or `roleOccupant` are treated as full DNs. Only direct users are supported. Binary group identifiers, nested groups and AD ranged membership responses are rejected.

LDAP is authoritative for the membership of every matched team, including manually added members. Retained members keep their team roles. New members receive the regular member role. The shared membership operation updates the team roster, user membership list and member budget/spend association together. Removing a member also deletes their existing keys for that team, following the same behavior as SCIM/team-member removal. User accounts and memberships in other teams are retained. Global user roles and disabled states are unchanged.

## Failure behavior

A group no longer returned by LDAP leaves its team and memberships unchanged. A successfully read group with no members removes all memberships from its corresponding team. No team or user is ever deleted, and no user is provisioned by this sync.

Failed or incomplete group searches, referrals, duplicate group IDs and invalid responses abort the directory snapshot without applying it. Missing, ambiguous, nested or unreadable user entries skip their affected group. A failure for one valid group does not prevent other valid groups from synchronizing. A user disappearing during the membership write causes that group's transaction to make no changes. Repeated successful runs with the same roster do not rewrite it.

The LDAP identity must have permission to read every selected group, its complete membership attributes, and the user identity attributes. LDAP can omit attributes hidden by access controls just as it omits empty attributes, so the sync cannot distinguish every permission restriction from an empty group. Verify the bind account and dry-run output before enabling writes.

Runs have a timeout shorter than their shared Redis lease. A timeout cancels further writes and signals the LDAP reader to stop between entries and lookups. LDAP socket timeouts and server search time limits use `timeout_seconds` as an upper bound and are capped further for short sync intervals, and the next interval retries from current state. Cancellation does not undo groups already committed earlier in the run. Stopping the sync (`enabled: false` or removing the block and restarting) leaves current teams and memberships in place.
