# insert data into clickhouse
# response = client.command(
#     """
#     CREATE TEMPORARY TABLE temp_spend_logs AS (
#     SELECT
#         generateUUIDv4() AS request_id,
#         arrayElement(['TypeA', 'TypeB', 'TypeC'], rand() % 3 + 1) AS call_type,
#         'ishaan' as api_key,
#         rand() * 1000 AS spend,
#         rand() * 100 AS total_tokens,
#         rand() * 50 AS prompt_tokens,
#         rand() * 50 AS completion_tokens,
#         toDate('2024-02-01') + toIntervalDay(rand()%27) AS startTime,
#         now() AS endTime,
#         arrayElement(['azure/gpt-4', 'gpt-3.5', 'vertexai/gemini-pro', 'mistral/mistral-small', 'ollama/llama2'], rand() % 3 + 1) AS model,
#         'ishaan-insert-rand' as user,
#         'data' as metadata,
#         'true'AS cache_hit,
#         'ishaan' as cache_key,
#         '{"tag1": "value1", "tag2": "value2"}' AS request_tags
#     FROM numbers(1, 1000000)
#     );
#     """
# )

# client.command(
#     """
#     -- Insert data into spend_logs table
#     INSERT INTO spend_logs
#     SELECT * FROM temp_spend_logs;
#     """
# )


# client.command(
#     """
#     DROP TABLE IF EXISTS temp_spend_logs;
#     """
# )
