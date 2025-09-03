import os
from typing import List, Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, status

from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


@router.get("/photon/config", dependencies=[Depends(user_api_key_auth)])
async def get_photon_config() -> dict:
    api_base = os.getenv("PHOTON_API_BASE", "").rstrip("/")
    models_path = os.getenv("PHOTON_MODELS_PATH", "/models")
    inference_base = f"{api_base}/inference"
    return {
        "api_base": api_base,
        "models_path": models_path,
        "inference_base": inference_base
    }



@router.get("/photon/models", dependencies=[Depends(user_api_key_auth)])
async def list_photon_models() -> dict:
    base = os.getenv("PHOTON_API_BASE", "").rstrip("/")
    if not base:
        # Misconfigured; return empty but 200 so UI can proceed gracefully
        return {"availableModels": []}
    
    url = f"{base}/models"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success") and isinstance(data.get("models"), list):
                    deployed_models = sorted(list(set([
                        model["modelSource"]
                        for model in data["models"]
                        if model.get("status") == "Deployed" and "modelSource" in model
                    ])))
                    return {"availableModels": deployed_models}
    except Exception:
        # If request fails, return empty array to prevent UI breakage
        pass

    # If all endpoints fail, return empty array to prevent UI breakage
    return {"availableModels": []}


