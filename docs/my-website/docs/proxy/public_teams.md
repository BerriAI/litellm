# [BETA] Public Teams

Expose available teams to your users to join on signup.

<iframe width="840" height="500" src="https://www.loom.com/embed/7871ea15035a48d2a118b7486c2f7598?sid=267cd0ab-d92b-42fa-b97a-9f385ef8930c" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>


## Quick Start

1. Create a team on LiteLLM

```bash
curl -X POST '<PROXY_BASE_URL>/team/new' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer <MASTER_KEY>' \
-d '{"name": "My Team", "team_id": "team_id_1"}'
```

2. Expose the team to your users

```yaml
litellm_settings:
    default_internal_user_params:
        available_teams: ["team_id_1"] # 👈 Make team available to new SSO users
```

3. Test it! 

```bash
curl -L -X POST 'http://0.0.0.0:4000/team/member_add' \
-H 'Authorization: Bearer sk-<USER_KEY>' \
-H 'Content-Type: application/json' \
--data-raw '{
    "team_id": "team_id_1", 
    "member": [{"role": "user", "user_id": "my-test-user"}]
}'
```



