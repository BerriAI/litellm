from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .mock_azure_batch import setup_batch_routes
from .mock_chat import setup_chat_routes
from .mock_embeddings import setup_embeddings_routes
from .mock_responses import setup_responses_routes
from .mock_s3_callback import setup_s3_callback_routes


def create_mock_azure_batch_server() -> FastAPI:
    """Create a FastAPI app that mocks Azure Batch API and S3 callbacks."""
    app = FastAPI()

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
    setup_s3_callback_routes(app)

    return app
