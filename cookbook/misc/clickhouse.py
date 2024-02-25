import clickhouse_connect

# client = clickhouse_connect.get_client(
#     host='hjs1z7j37j.us-east1.gcp.clickhouse.cloud',
#     port=8443,
#     username='default',
#     password='M~PimRs~c3Z6b',
#     secure=False
# )
import clickhouse_connect

client = clickhouse_connect.get_client()


row1 = [
    "123456",  # request_id
    "GET",  # call_type
    "api_key_123",  # api_key
    50.00,  # spend
    1000,  # total_tokens
    800,  # prompt_tokens
    200,  # completion_tokens
    "2024-02-24 12:00",  # startTime (replace with the actual timestamp)
    "2024-02-24 13:00",  # endTime (replace with the actual timestamp)
    "gpt-3.5",  # model
    "user123",  # user
    '{"key": "value"}',  # metadata (replace with valid JSON)
    True,  # cache_hit
    "cache_key_123",  # cache_key
    "tag1,tag2",  # request_tags
]

row2 = [
    "789012",  # request_id
    "POST",  # call_type
    "api_key_456",  # api_key
    30.50,  # spend
    800,  # total_tokens
    600,  # prompt_tokens
    200,  # completion_tokens
    "2024-02-24 14:00",  # startTime (replace with the actual timestamp)
    "2024-02-24 15:00",  # endTime (replace with the actual timestamp)
    "gpt-4.0",  # model
    "user456",  # user
    '{"key": "value"}',  # metadata (replace with valid JSON)
    False,  # cache_hit
    "cache_key_789",  # cache_key
    "tag3,tag4",  # request_tags
]

data = [row1, row2]
client.insert(
    "spend_logs",
    data,
    column_names=[
        "request_id",
        "call_type",
        "api_key",
        "spend",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "startTime",
        "endTime",
        "model",
        "user",
        "metadata",
        "cache_hit",
        "cache_key",
        "request_tags",
    ],
)
