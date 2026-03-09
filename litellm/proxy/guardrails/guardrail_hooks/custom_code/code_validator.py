import re
from typing import List, Tuple

# Security validation patterns
FORBIDDEN_PATTERNS: List[Tuple[str, str]] = [
    # Import statements
    (r"\bimport\s+", "import statements are not allowed"),
    (r"\bfrom\s+\w+\s+import\b", "from...import statements are not allowed"),
    (r"__import__\s*\(", "__import__() is not allowed"),
    # Dangerous builtins
    (r"\bexec\s*\(", "exec() is not allowed"),
    (r"\beval\s*\(", "eval() is not allowed"),
    (r"\bcompile\s*\(", "compile() is not allowed"),
    (r"\bopen\s*\(", "open() is not allowed"),
    (r"\bgetattr\s*\(", "getattr() is not allowed"),
    (r"\bsetattr\s*\(", "setattr() is not allowed"),
    (r"\bdelattr\s*\(", "delattr() is not allowed"),
    (r"\bglobals\s*\(", "globals() is not allowed"),
    (r"\blocals\s*\(", "locals() is not allowed"),
    (r"\bvars\s*\(", "vars() is not allowed"),
    (r"\bdir\s*\(", "dir() is not allowed"),
    (r"\bbreakpoint\s*\(", "breakpoint() is not allowed"),
    (r"\binput\s*\(", "input() is not allowed"),
    # Dangerous dunder access
    (r"__builtins__", "__builtins__ access is not allowed"),
    (r"__globals__", "__globals__ access is not allowed"),
    (r"__code__", "__code__ access is not allowed"),
    (r"__subclasses__", "__subclasses__ access is not allowed"),
    (r"__bases__", "__bases__ access is not allowed"),
    (r"__mro__", "__mro__ access is not allowed"),
    (r"__class__", "__class__ access is not allowed"),
    (r"__dict__", "__dict__ access is not allowed"),
    (r"__getattribute__", "__getattribute__ access is not allowed"),
    (r"__reduce__", "__reduce__ access is not allowed"),
    (r"__reduce_ex__", "__reduce_ex__ access is not allowed"),
    # OS/system access
    (r"\bos\.", "os module access is not allowed"),
    (r"\bsys\.", "sys module access is not allowed"),
    (r"\bsubprocess\.", "subprocess module access is not allowed"),
    (r"\bshutil\.", "shutil module access is not allowed"),
    (r"\bctypes\.", "ctypes module access is not allowed"),
    (r"\bsocket\.", "socket module access is not allowed"),
    (r"\bpickle\.", "pickle module access is not allowed"),
]


class CustomCodeValidationError(Exception):
    """Raised when custom code fails security validation."""

    pass


def validate_custom_code(code: str) -> None:
    """
    Validate custom code against forbidden patterns.

    Raises CustomCodeValidationError if any forbidden pattern is found.
    """
    if not code:
        return
    for pattern, error_msg in FORBIDDEN_PATTERNS:
        if re.search(pattern, code):
            raise CustomCodeValidationError(f"Security violation: {error_msg}")
