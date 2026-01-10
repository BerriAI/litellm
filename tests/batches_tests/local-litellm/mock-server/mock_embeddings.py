from fastapi import FastAPI, Request


def setup_embeddings_routes(app: FastAPI):
    @app.post("/embeddings")
    @app.post("/v1/embeddings")
    @app.post("/openai/deployments/{model:path}/embeddings")
    async def embeddings(request: Request):
        data = await request.json()
        model = data.get("model", "unknown")
        _small_embedding = [
            -0.006929283495992422,
            -0.005336422007530928,
            -4.547132266452536e-05,
            -0.024047505110502243,
        ]
        big_embedding = _small_embedding * 100
        return {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": big_embedding}],
            "model": model,
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }



