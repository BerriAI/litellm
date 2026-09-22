from collections.abc import Collection


def is_fastapi_http_exception(e: Exception, block_status_codes: Collection[int]) -> bool:
    try:
        from fastapi.exceptions import HTTPException
    except ImportError:
        return False
    return isinstance(e, HTTPException) and e.status_code in block_status_codes
