class BlockingException(BaseException):
    """
    Exception raised when a request is blocked by ASM
    It derives from BaseException to avoid being caught by the general Exception handler
    """
