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