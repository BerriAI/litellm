import ast
import re
from typing import List, Set, Tuple

from litellm._logging import verbose_proxy_logger

# Security validation patterns (regex-based, first layer of defense)
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

# Dangerous function names that should not appear as calls in the AST.
# NOTE: Common Python constructs (type, super, classmethod, staticmethod,
# property, object) are intentionally NOT blocked here — guardrail code may
# legitimately use them, and the empty __builtins__ sandbox already prevents
# their abuse at runtime.
FORBIDDEN_CALL_NAMES: Set[str] = {
    "exec",
    "eval",
    "compile",
    "open",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "dir",
    "breakpoint",
    "input",
    "__import__",
    "memoryview",
}

# Dangerous attribute names that should not be accessed in the AST
FORBIDDEN_ATTR_NAMES: Set[str] = {
    "__builtins__",
    "__globals__",
    "__code__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__class__",
    "__dict__",
    "__getattribute__",
    "__reduce__",
    "__reduce_ex__",
    "__loader__",
    "__spec__",
    "__qualname__",
    "__module__",
    "__init_subclass__",
    "__set_name__",
    "__wrapped__",
    "__func__",
    "__self__",
    "__closure__",
    # Dangerous dunder methods that can be used for sandbox escape
    "__init__",
    "__new__",
    # f-string internals
    "format_map",
}


class CustomCodeValidationError(Exception):
    """Raised when custom code fails security validation."""

    pass


class _ASTSecurityVisitor(ast.NodeVisitor):
    """
    AST visitor that detects dangerous constructs that regex cannot catch.

    This provides defense-in-depth against sandbox escape attempts such as:
    - String concatenation to construct forbidden names at runtime
    - Using globals()['__import__'] style access
    - Attribute chains like ().__class__.__bases__[0].__subclasses__()
    """

    def __init__(self) -> None:
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        self.violations.append("import statements are not allowed")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.violations.append("from...import statements are not allowed")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Check direct function calls: exec(...), eval(...)
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALL_NAMES:
                self.violations.append(f"{node.func.id}() is not allowed")
        # Note: attribute-based calls (e.g. obj.__subclasses__()) are already
        # caught by visit_Attribute, so no need for a duplicate check here.
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in FORBIDDEN_ATTR_NAMES:
            self.violations.append(f"{node.attr} access is not allowed")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.violations.append("global statement is not allowed")
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        # Nonlocal is fine for closures within the guardrail code
        self.generic_visit(node)


def _validate_ast(code: str) -> None:
    """
    Validate custom code using AST analysis (second layer of defense).

    Parses the code into an AST and walks it to detect dangerous constructs
    that regex-based validation might miss (e.g., string concatenation tricks,
    dynamic attribute access patterns).

    Raises CustomCodeValidationError if any dangerous construct is found.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Syntax errors will be caught later during compile(); not a security issue
        return

    visitor = _ASTSecurityVisitor()
    visitor.visit(tree)

    if visitor.violations:
        # De-duplicate while preserving order, then report all violations
        seen: set[str] = set()
        unique: list[str] = []
        for v in visitor.violations:
            if v not in seen:
                seen.add(v)
                unique.append(v)
        summary = "; ".join(unique)
        raise CustomCodeValidationError(f"Security violation(s): {summary}")


def validate_custom_code(code: str) -> None:
    """
    Validate custom code against forbidden patterns using layered defense.

    Layer 1: Regex-based pattern matching (catches obvious violations)
    Layer 2: AST-based analysis (catches evasion attempts)

    Raises CustomCodeValidationError if any forbidden pattern is found.
    """
    if not code:
        return

    # Layer 1: Regex-based validation
    for pattern, error_msg in FORBIDDEN_PATTERNS:
        if re.search(pattern, code):
            raise CustomCodeValidationError(f"Security violation: {error_msg}")

    # Layer 2: AST-based validation (defense-in-depth)
    _validate_ast(code)

    verbose_proxy_logger.debug("Custom code passed security validation (regex + AST)")
