from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from mock_azure_batch import setup_batch_routes
from mock_chat import setup_chat_routes
from mock_embeddings import setup_embeddings_routes
from mock_responses import setup_responses_routes


def get_request_url(request: Request):
    return str(request.url)


limiter = Limiter(key_func=get_request_url)
load_dotenv()

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


setup_chat_routes(app)
setup_responses_routes(app)
setup_embeddings_routes(app)
setup_batch_routes(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8090)
