from fastapi import APIRouter
from fastapi.responses import Response
from .utils import _should_block_robots

router = APIRouter()


@router.get("/robots.txt")
async def get_robots():
    """
    Block all web crawlers from indexing the proxy server endpoints
    This is useful for ensuring that the API endpoints aren't indexed by search engines
    """
    if _should_block_robots():
        return Response(content="User-agent: *\nDisallow: /", media_type="text/plain")
    else:
        return Response(status_code=404)