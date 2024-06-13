import Image from '@theme/IdealImage';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Managing Team Budgets

Track spend, set budgets for your Internal Team

## Setting Monthly Team Budgets

### 1. Create a team 
with `max_budget` and `budget_duration` 

<Tabs>
<TabItem value="api" label="API">

Set `max_budget` and `budget_duration`
```shell
curl --location 'http://0.0.0.0:4000/team/new' \
     --header 'Authorization: Bearer sk-1234' \
     --header 'Content-Type: application/json' \
     --data '{
         "team_alias": "QA Prod Bot",
         "max_budget": 0.000000001,
         "budget_duration": "1d"
     }'   
```

Response
```shell
{
 "team_alias": "QA Prod Bot",
 "team_id": "de35b29e-6ca8-4f47-b804-2b79d07aa99a",
 "max_budget": 0.0001,
 "budget_duration": "1d",
 "budget_reset_at": "2024-06-14T22:48:36.594000Z"
}  
```
</TabItem>
<TabItem value="ui" label="UI">
</TabItem>
</Tabs>

### 2. Create a key for the `team`

```shell
curl --location 'http://0.0.0.0:4000/key/generate' \
     --header 'Authorization: Bearer sk-1234' \
     --header 'Content-Type: application/json' \
     --data '{
         "team_id": "de35b29e-6ca8-4f47-b804-2b79d07aa99a"
     }'
```

Response

```shell
{"team_id":"de35b29e-6ca8-4f47-b804-2b79d07aa99a", "key":"sk-5qtncoYjzRcxMM4bDRktNQ"}
```


### 3. Test It

Run this Request twice
```shell
curl --location 'http://0.0.0.0:4000/chat/completions' \
--header 'Authorization: Bearer sk-mso-JSykEGri86KyOvgxBw' \
--header 'Content-Type: application/json' \
--data ' {
      "model": "llama3",
      "messages": [
        {
          "role": "user",
          "content": "hi"
        }
      ],
    }
'
```

On the 2nd response - expect to see the following exception

```shell
{
 "error": {
   "message": "Budget has been exceeded! Current cost: 3.5e-06, Max budget: 1e-09",
   "type": "auth_error",
   "param": null,
   "code": 400
 }
}
```


### 4. Prometheus metrics for `remaining_budget`


## Updating Team Budgets