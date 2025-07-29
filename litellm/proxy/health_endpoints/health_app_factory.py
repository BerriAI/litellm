from fastapi import FastAPI
from litellm.proxy.health_endpoints._health_endpoints import router as health_router

def build_health_app():
    health_app = FastAPI(title="LiteLLM Health Endpoints")
    health_app.include_router(health_router)
    return health_app 