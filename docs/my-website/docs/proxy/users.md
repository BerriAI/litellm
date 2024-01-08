# 💰 Budgets, Rate Limits per user 

Requirements: 

- Need to a postgres database (e.g. [Supabase](https://supabase.com/), [Neon](https://neon.tech/), etc)


## Set Budgets
LiteLLM exposes a `/user/new` endpoint to create budgets for users, that persist across multiple keys. 

This is documented in the swagger (live on your server root endpoint - e.g. `http://0.0.0.0:8000/`). Here's an example request. 

```shell 
curl --location 'http://localhost:8000/user/new' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{"models": ["azure-models"], "max_budget": 0, "user_id": "krrish3@berri.ai"}' 
```
The request is a normal `/key/generate` request body + a `max_budget` field. 

**Sample Response**

```shell
{
    "key": "sk-YF2OxDbrgd1y2KgwxmEA2w",
    "expires": "2023-12-22T09:53:13.861000Z",
    "user_id": "krrish3@berri.ai",
    "max_budget": 0.0
}
```


## Set Rate Limits 

Set max parallel requests a user can make, when you create user keys - `/key/generate`. 

```shell
curl --location 'http://0.0.0.0:8000/key/generate' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{"duration": "20m", "max_parallel_requests": 1}' # 👈 max parallel requests = 1
```