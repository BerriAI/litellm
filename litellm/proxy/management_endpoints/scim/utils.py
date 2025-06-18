from fastapi import HTTPException


async def _check_user_exists(prisma_client, user_name: str) -> bool:
    """Check if user already exists by username"""
    if not user_name:
        return False
    
    existing_user = await prisma_client.db.litellm_usertable.find_unique(
        where={"user_id": user_name}
    )
    return existing_user is not None


def _extract_error_message(http_exception: HTTPException) -> str:
    """Extract error message from HTTPException detail"""
    if isinstance(http_exception.detail, dict):
        return http_exception.detail.get("error", "User already exists")
    return str(http_exception.detail)

