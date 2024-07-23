import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 👥📊 Team Based Logging

Allow each team to use their own Langfuse Project / custom callbacks

**This allows you to do the following**
```
Team 1 -> Logs to Langfuse Project 1 
Team 2 -> Logs to Langfuse Project 2
Team 3 -> Disabled Logging (for GDPR compliance)
```

## Set Callbacks Per Team

### 1. Set callback for team 

We make a request to `POST /team/{team_id}/callback` to add a callback for

```shell
curl -X POST 'http:/localhost:4000/team/dbe2f686-a686-4896-864a-4c3924458709/callback' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-d '{
  "callback_name": "langfuse",
  "callback_type": "success",
  "callback_vars": {
    "langfuse_public_key": "pk", 
    "langfuse_secret_key": "sk_", 
    "langfuse_host": "https://cloud.langfuse.com"
    }
  
}'
```

#### Supported Values

| Field | Supported Values | Notes |
|-------|------------------|-------|
| `callback_name` | `"langfuse"` | Currently only supports "langfuse" |
| `callback_type` | `"success"`, `"failure"`, `"success_and_failure"` | |
| `callback_vars` | | dict of callback settings |
| &nbsp;&nbsp;&nbsp;&nbsp;`langfuse_public_key` | string | Required |
| &nbsp;&nbsp;&nbsp;&nbsp;`langfuse_secret_key` | string | Required |
| &nbsp;&nbsp;&nbsp;&nbsp;`langfuse_host` | string | Optional (defaults to https://cloud.langfuse.com) |

### 2. Create key for team

All keys created for team `dbe2f686-a686-4896-864a-4c3924458709` will log to langfuse project specified on [Step 1. Set callback for team](#1-set-callback-for-team)


```shell
curl --location 'http://0.0.0.0:4000/key/generate' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data '{
        "team_id": "dbe2f686-a686-4896-864a-4c3924458709"
}'
```


### 3. Make `/chat/completion` request for team

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-KbUuE0WNptC0jXapyMmLBA" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Hello, Claude gm!"}
    ]
}'
```

Expect this to be logged on the langfuse project specified on [Step 1. Set callback for team](#1-set-callback-for-team)


## Disable Logging for a Team

To disable logging for a specific team, you can use the following endpoint:

`POST /team/{team_id}/disable_logging`

This endpoint removes all success and failure callbacks for the specified team, effectively disabling logging.

### Step 1. Disable logging for team

```shell
curl -X POST 'http://localhost:4000/team/YOUR_TEAM_ID/disable_logging' \
    -H 'Authorization: Bearer YOUR_API_KEY'
```
Replace YOUR_TEAM_ID with the actual team ID

**Response**
A successful request will return a response similar to this:
```json
{
    "status": "success",
    "message": "Logging disabled for team YOUR_TEAM_ID",
    "data": {
        "team_id": "YOUR_TEAM_ID",
        "success_callbacks": [],
        "failure_callbacks": []
    }
}
```

### Step 2. Test it - `/chat/completions`

Use a key generated for team = `team_id` - you should see no logs on your configured success callback (eg. Langfuse)

```shell
curl -i http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-KbUuE0WNptC0jXapyMmLBA" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Hello, Claude gm!"}
    ]
}'
```

### Debugging / Troubleshooting

- Check active callbacks for team using `GET /team/{team_id}/callback`

Use this to check what success/failure callbacks are active for team=`team_id`

```shell
curl -X GET 'http://localhost:4000/team/dbe2f686-a686-4896-864a-4c3924458709/callback' \
        -H 'Authorization: Bearer sk-1234'
```

## Team Logging Endpoints

- [`POST /team/{team_id}/callback` Add a success/failure callback to a team](https://litellm-api.up.railway.app/#/team%20management/add_team_callbacks_team__team_id__callback_post)
- [`GET /team/{team_id}/callback` - Get the success/failure callbacks and variables for a team](https://litellm-api.up.railway.app/#/team%20management/get_team_callbacks_team__team_id__callback_get)



