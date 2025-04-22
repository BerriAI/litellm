# 💸 GET Daily Spend, Usage Metrics

## Request Format
```shell
curl -X GET "http://0.0.0.0:4000/daily_metrics" -H "Authorization: Bearer sk-1234"
```

## Response format 
```json
[
    daily_spend = [
        {
            "daily_spend": 7.9261938052047e+16,
            "day": "2024-02-01T00:00:00",
            "spend_per_model": {"azure/gpt-4": 7.9261938052047e+16},
            "spend_per_api_key": {
                "76": 914495704992000.0,
                "12": 905726697912000.0,
                "71": 866312628003000.0,
                "28": 865461799332000.0,
                "13": 859151538396000.0
            }
        },
        {
            "daily_spend": 7.938489251309491e+16,
            "day": "2024-02-02T00:00:00",
            "spend_per_model": {"gpt-3.5": 7.938489251309491e+16},
            "spend_per_api_key": {
                "91": 896805036036000.0,
                "78": 889692646082000.0,
                "49": 885386687861000.0,
                "28": 873869890984000.0,
                "56": 867398637692000.0
            }
        }

    ],
    total_spend = 200,
    top_models = {"gpt4": 0.2, "vertexai/gemini-pro":10},
    top_api_keys = {"899922": 0.9, "838hcjd999seerr88": 20}

]

```