import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# 🙋‍♂️ Customers 

Track spend, set budgets for your customers.

## Tracking Customer Credit

1. Track LLM API Spend by Customer ID

Make a /chat/completions call, pass 'user' - First call Works

```bash
curl -X POST 'http://0.0.0.0:4000/chat/completions' \
        --header 'Content-Type: application/json' \
        --header 'Authorization: Bearer sk-1234' \ # 👈 YOUR PROXY KEY
        --data ' {
        "model": "azure-gpt-3.5",
        "user": "ishaan3", # 👈 CUSTOMER ID
        "messages": [
            {
            "role": "user",
            "content": "what time is it"
            }
        ]
        }'
```

The customer_id will be upserted into the DB with the new spend.

If the customer_id already exists, spend will be incremented.

2. Get Customer Spend 

<Tabs>
<TabItem value="all-up" label="All-up spend">

Call `/customer/info` to get a customer's all up spend

```bash
curl -X GET 'http://0.0.0.0:4000/customer/info?end_user_id=ishaan3' \ # 👈 CUSTOMER ID
        -H 'Authorization: Bearer sk-1234' \ # 👈 YOUR PROXY KEY
```

Expected Response:

```
{
    "user_id": "ishaan3",
    "blocked": false,
    "alias": null,
    "spend": 0.001413,
    "allowed_model_region": null,
    "default_model": null,
    "litellm_budget_table": null
}
```

</TabItem>
<TabItem value="event-webhook" label="Event Webhook">

To update spend in your client-side DB, point the proxy to your webhook. 

E.g. if your server is `https://webhook.site` and your listening on `6ab090e8-c55f-4a23-b075-3209f5c57906`

1. Add webhook url to your proxy environment: 

```bash
export WEBHOOK_URL="https://webhook.site/6ab090e8-c55f-4a23-b075-3209f5c57906"
```

2. Add 'webhook' to config.yaml

```yaml
general_settings: 
  alerting: ["webhook"] # 👈 KEY CHANGE
```

3. Test it! 

```bash
curl -X POST 'http://localhost:4000/chat/completions' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer sk-1234' \
-D '{
    "model": "mistral",
    "messages": [
        {
        "role": "user",
        "content": "What's the weather like in Boston today?"
        }
    ],
    "user": "krrish12"
}
'
```

Expected Response 

```json
{
  "spend": 0.0011120000000000001, # 👈 SPEND
  "max_budget": null,
  "token": "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
  "customer_id": "krrish12",  # 👈 CUSTOMER ID
  "user_id": null,
  "team_id": null,
  "user_email": null,
  "key_alias": null,
  "projected_exceeded_date": null,
  "projected_spend": null,
  "event": "spend_tracked",
  "event_group": "customer",
  "event_message": "Customer spend tracked. Customer=krrish12, spend=0.0011120000000000001"
}
```

[See Webhook Spec](./alerting.md#api-spec-for-webhook-event)

</TabItem>
</Tabs>
