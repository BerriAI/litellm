import copy

from fastapi import APIRouter, Depends, HTTPException, Request

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import RedisCache
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter(
    prefix="/cache",
    tags=["caching"],
)


@router.get(
    "/ping",
    dependencies=[Depends(user_api_key_auth)],
)
async def cache_ping():
    """
    Endpoint for checking if cache can be pinged
    """
    litellm_cache_params = {}
    specific_cache_params = {}
    try:

        if litellm.cache is None:
            raise HTTPException(
                status_code=503, detail="Cache not initialized. litellm.cache is None"
            )

        for k, v in vars(litellm.cache).items():
            try:
                if k == "cache":
                    continue
                litellm_cache_params[k] = str(copy.deepcopy(v))
            except Exception:
                litellm_cache_params[k] = "<unable to copy or convert>"
        for k, v in vars(litellm.cache.cache).items():
            try:
                specific_cache_params[k] = str(v)
            except Exception:
                specific_cache_params[k] = "<unable to copy or convert>"
        if litellm.cache.type == "redis":
            # ping the redis cache
            ping_response = await litellm.cache.ping()
            verbose_proxy_logger.debug(
                "/cache/ping: ping_response: " + str(ping_response)
            )
            # making a set cache call
            # add cache does not return anything
            await litellm.cache.async_add_cache(
                result="test_key",
                model="test-model",
                messages=[{"role": "user", "content": "test from litellm"}],
            )
            verbose_proxy_logger.debug("/cache/ping: done with set_cache()")
            return {
                "status": "healthy",
                "cache_type": litellm.cache.type,
                "ping_response": True,
                "set_cache_response": "success",
                "litellm_cache_params": litellm_cache_params,
                "redis_cache_params": specific_cache_params,
            }
        else:
            return {
                "status": "healthy",
                "cache_type": litellm.cache.type,
                "litellm_cache_params": litellm_cache_params,
            }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service Unhealthy ({str(e)}).Cache parameters: {litellm_cache_params}.specific_cache_params: {specific_cache_params}",
        )


@router.post(
    "/delete",
    tags=["caching"],
    dependencies=[Depends(user_api_key_auth)],
)
async def cache_delete(request: Request):
    """
    Endpoint for deleting a key from the cache. All responses from litellm proxy have `x-litellm-cache-key` in the headers

    Parameters:
    - **keys**: *Optional[List[str]]* - A list of keys to delete from the cache. Example {"keys": ["key1", "key2"]}

    ```shell
    curl -X POST "http://0.0.0.0:4000/cache/delete" \
    -H "Authorization: Bearer sk-1234" \
    -d '{"keys": ["key1", "key2"]}'
    ```

    """
    try:
        if litellm.cache is None:
            raise HTTPException(
                status_code=503, detail="Cache not initialized. litellm.cache is None"
            )

        request_data = await request.json()
        keys = request_data.get("keys", None)

        if litellm.cache.type == "redis":
            await litellm.cache.delete_cache_keys(keys=keys)
            return {
                "status": "success",
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Cache type {litellm.cache.type} does not support deleting a key. only `redis` is supported",
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cache Delete Failed({str(e)})",
        )


@router.get(
    "/redis/info",
    dependencies=[Depends(user_api_key_auth)],
)
async def cache_redis_info():
    """
    Endpoint for getting /redis/info
    """
    try:
        if litellm.cache is None:
            raise HTTPException(
                status_code=503, detail="Cache not initialized. litellm.cache is None"
            )
        if litellm.cache.type == "redis" and isinstance(
            litellm.cache.cache, RedisCache
        ):
            client_list = litellm.cache.cache.client_list()
            redis_info = litellm.cache.cache.info()
            num_clients = len(client_list)
            return {
                "num_clients": num_clients,
                "clients": client_list,
                "info": redis_info,
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Cache type {litellm.cache.type} does not support flushing",
            )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service Unhealthy ({str(e)})",
        )


@router.post(
    "/flushall",
    tags=["caching"],
    dependencies=[Depends(user_api_key_auth)],
)
async def cache_flushall():
    """
    A function to flush all items from the cache. (All items will be deleted from the cache with this)
    Raises HTTPException if the cache is not initialized or if the cache type does not support flushing.
    Returns a dictionary with the status of the operation.

    Usage:
    ```
    curl -X POST http://0.0.0.0:4000/cache/flushall -H "Authorization: Bearer sk-1234"
    ```
    """
    try:
        if litellm.cache is None:
            raise HTTPException(
                status_code=503, detail="Cache not initialized. litellm.cache is None"
            )
        if litellm.cache.type == "redis" and isinstance(
            litellm.cache.cache, RedisCache
        ):
            litellm.cache.cache.flushall()
            return {
                "status": "success",
            }
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Cache type {litellm.cache.type} does not support flushing",
            )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Service Unhealthy ({str(e)})",
        )
