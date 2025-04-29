from fastapi import APIRouter
from fastapi.responses import Response
from litellm.secret_managers.main import str_to_bool

router = APIRouter()

def _should_block_robots():
    from litellm.proxy.proxy_server import general_settings, premium_user, CommonProxyErrors
    _block_robots = general_settings.get("block_robots", False)
    if str_to_bool(_block_robots) is True:
        if premium_user is not True:
            raise ValueError(f"Blocking web crawlers is an enterprise feature. {CommonProxyErrors.not_premium_user.value}")
        return True
    return False

if _should_block_robots():
    @router.get("/robots.txt")
    async def get_robots():
        """
        Block all web crawlers from indexing the proxy server endpoints
        This is useful for ensuring that the API endpoints aren't indexed by search engines
        """
        return Response(content="User-agent: *\nDisallow: /", media_type="text/plain")