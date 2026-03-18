"""
Built-in primitives provided to custom code guardrails.

These functions are injected into the custom code execution environment
and provide safe, sandboxed functionality for common guardrail operations.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from urllib.parse import urlparse

import httpx

from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider

# =============================================================================
# Result Types - Used by Starlark code to return guardrail decisions
# =============================================================================


def allow() -> Dict[str, Any]:
    """
    Allow the request/response to proceed unchanged.

    Returns:
        Dict indicating the request should be allowed
    """
    return {"action": "allow"}


def block(
    reason: str, detection_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Block the request/response with a reason.

    Args:
        reason: Human-readable reason for blocking
        detection_info: Optional additional detection metadata

    Returns:
        Dict indicating the request should be blocked
    """
    result: Dict[str, Any] = {"action": "block", "reason": reason}
    if detection_info:
        result["detection_info"] = detection_info
    return result


def modify(
    texts: Optional[List[str]] = None,
    images: Optional[List[Any]] = None,
    tool_calls: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Modify the request/response content.

    Args:
        texts: Modified text content (if None, keeps original)
        images: Modified image content (if None, keeps original)
        tool_calls: Modified tool calls (if None, keeps original)

    Returns:
        Dict indicating the content should be modified
    """
    result: Dict[str, Any] = {"action": "modify"}
    if texts is not None:
        result["texts"] = texts
    if images is not None:
        result["images"] = images
    if tool_calls is not None:
        result["tool_calls"] = tool_calls
    return result


# =============================================================================
# Regex Primitives
# =============================================================================


def regex_match(text: str, pattern: str, flags: int = 0) -> bool:
    """
    Check if a regex pattern matches anywhere in the text.

    Args:
        text: The text to search in
        pattern: The regex pattern to match
        flags: Optional regex flags (default: 0)

    Returns:
        True if pattern matches, False otherwise
    """
    try:
        return bool(re.search(pattern, text, flags))
    except re.error as e:
        verbose_proxy_logger.warning(f"Starlark regex_match error: {e}")
        return False


def regex_match_all(text: str, pattern: str, flags: int = 0) -> bool:
    """
    Check if a regex pattern matches the entire text.

    Args:
        text: The text to match
        pattern: The regex pattern
        flags: Optional regex flags

    Returns:
        True if pattern matches entire text, False otherwise
    """
    try:
        return bool(re.fullmatch(pattern, text, flags))
    except re.error as e:
        verbose_proxy_logger.warning(f"Starlark regex_match_all error: {e}")
        return False


def regex_replace(text: str, pattern: str, replacement: str, flags: int = 0) -> str:
    """
    Replace all occurrences of a pattern in text.

    Args:
        text: The text to modify
        pattern: The regex pattern to find
        replacement: The replacement string
        flags: Optional regex flags

    Returns:
        The text with replacements applied
    """
    try:
        return re.sub(pattern, replacement, text, flags=flags)
    except re.error as e:
        verbose_proxy_logger.warning(f"Starlark regex_replace error: {e}")
        return text


def regex_find_all(text: str, pattern: str, flags: int = 0) -> List[str]:
    """
    Find all occurrences of a pattern in text.

    Args:
        text: The text to search
        pattern: The regex pattern to find
        flags: Optional regex flags

    Returns:
        List of all matches
    """
    try:
        return re.findall(pattern, text, flags)
    except re.error as e:
        verbose_proxy_logger.warning(f"Starlark regex_find_all error: {e}")
        return []


# =============================================================================
# JSON Primitives
# =============================================================================


def json_parse(text: str) -> Optional[Any]:
    """
    Parse a JSON string into a Python object.

    Args:
        text: The JSON string to parse

    Returns:
        Parsed Python object, or None if parsing fails
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as e:
        verbose_proxy_logger.debug(f"Starlark json_parse error: {e}")
        return None


def json_stringify(obj: Any) -> str:
    """
    Convert a Python object to a JSON string.

    Args:
        obj: The object to serialize

    Returns:
        JSON string representation
    """
    try:
        return json.dumps(obj)
    except (TypeError, ValueError) as e:
        verbose_proxy_logger.warning(f"Starlark json_stringify error: {e}")
        return ""


def json_schema_valid(obj: Any, schema: Dict[str, Any]) -> bool:
    """
    Validate an object against a JSON schema.

    Args:
        obj: The object to validate
        schema: The JSON schema to validate against

    Returns:
        True if valid, False otherwise
    """
    try:
        # Try to import jsonschema, fall back to basic validation if not available
        try:
            import jsonschema

            jsonschema.validate(instance=obj, schema=schema)
            return True
        except ImportError:
            # Basic validation without jsonschema library
            return _basic_json_schema_validate(obj, schema)
        except Exception as validation_error:
            # Catch jsonschema.ValidationError and other validation errors
            if "ValidationError" in type(validation_error).__name__:
                return False
            raise
    except Exception as e:
        verbose_proxy_logger.warning(f"Custom code json_schema_valid error: {e}")
        return False


def _basic_json_schema_validate(
    obj: Any, schema: Dict[str, Any], max_depth: int = 50
) -> bool:
    """
    Basic JSON schema validation without external library.
    Handles: type, required, properties

    Uses an iterative approach with a stack to avoid recursion limits.
    max_depth limits nesting to prevent infinite loops from circular schemas.
    """
    type_map: Dict[str, Union[Type, Tuple[Type, ...]]] = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "null": type(None),
    }

    # Stack of (obj, schema, depth) tuples to process
    stack: List[Tuple[Any, Dict[str, Any], int]] = [(obj, schema, 0)]

    while stack:
        current_obj, current_schema, depth = stack.pop()

        # Circuit breaker: stop if we've gone too deep
        if depth > max_depth:
            return False

        # Check type
        schema_type = current_schema.get("type")
        if schema_type:
            expected_type = type_map.get(schema_type)
            if expected_type is not None and not isinstance(current_obj, expected_type):
                return False

        # Check required fields and properties for dicts
        if isinstance(current_obj, dict):
            required = current_schema.get("required", [])
            for field in required:
                if field not in current_obj:
                    return False

            # Queue property validations
            properties = current_schema.get("properties", {})
            for prop_name, prop_schema in properties.items():
                if prop_name in current_obj:
                    stack.append((current_obj[prop_name], prop_schema, depth + 1))

    return True


# =============================================================================
# URL Primitives
# =============================================================================


# Common URL pattern for extraction
_URL_PATTERN = re.compile(
    r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*", re.IGNORECASE
)


def extract_urls(text: str) -> List[str]:
    """
    Extract all URLs from text.

    Args:
        text: The text to search for URLs

    Returns:
        List of URLs found in the text
    """
    return _URL_PATTERN.findall(text)


def is_valid_url(url: str) -> bool:
    """
    Check if a URL is syntactically valid.

    Args:
        url: The URL to validate

    Returns:
        True if the URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def all_urls_valid(text: str) -> bool:
    """
    Check if all URLs in text are valid.

    Args:
        text: The text containing URLs

    Returns:
        True if all URLs are valid (or no URLs), False otherwise
    """
    urls = extract_urls(text)
    return all(is_valid_url(url) for url in urls)


def get_url_domain(url: str) -> Optional[str]:
    """
    Extract the domain from a URL.

    Args:
        url: The URL to parse

    Returns:
        The domain, or None if invalid
    """
    try:
        result = urlparse(url)
        return result.netloc if result.netloc else None
    except Exception:
        return None


# =============================================================================
# HTTP Request Primitives (Async)
# =============================================================================

# Default timeout for HTTP requests (in seconds)
_HTTP_DEFAULT_TIMEOUT = 30.0

# Maximum allowed timeout (in seconds)
_HTTP_MAX_TIMEOUT = 60.0


def _http_error_response(error: str) -> Dict[str, Any]:
    """Create a standardized error response for HTTP requests."""
    return {
        "status_code": 0,
        "body": None,
        "headers": {},
        "success": False,
        "error": error,
    }


def _http_success_response(response: httpx.Response) -> Dict[str, Any]:
    """Create a standardized success response from an httpx Response."""
    parsed_body: Any
    try:
        parsed_body = response.json()
    except (json.JSONDecodeError, ValueError):
        parsed_body = response.text

    return {
        "status_code": response.status_code,
        "body": parsed_body,
        "headers": dict(response.headers),
        "success": 200 <= response.status_code < 300,
        "error": None,
    }


def _prepare_http_body(
    body: Optional[Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Prepare body arguments for HTTP request - returns (json_body, data_body)."""
    if body is None:
        return None, None
    if isinstance(body, dict):
        return body, None
    if isinstance(body, list):
        return None, json.dumps(body)
    if isinstance(body, str):
        return None, body
    return None, str(body)


async def http_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    body: Optional[Any] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Make an async HTTP request to an external service.

    This function allows custom guardrails to call external APIs for
    additional validation, content moderation, or data enrichment.

    Uses LiteLLM's global cached AsyncHTTPHandler for connection pooling
    and better performance.

    Args:
        url: The URL to request
        method: HTTP method (GET, POST, PUT, DELETE, PATCH). Defaults to GET.
        headers: Optional dict of HTTP headers
        body: Optional request body (will be JSON-encoded if dict/list)
        timeout: Optional timeout in seconds (default: 30, max: 60)

    Returns:
        Dict containing:
            - status_code: HTTP status code
            - body: Response body (parsed as JSON if possible, otherwise string)
            - headers: Response headers as dict
            - success: True if status code is 2xx
            - error: Error message if request failed, None otherwise

    Example:
        # Simple GET request
        response = await http_request("https://api.example.com/check")
        if response["success"]:
            data = response["body"]

        # POST request with JSON body
        response = await http_request(
            "https://api.example.com/moderate",
            method="POST",
            headers={"Authorization": "Bearer token"},
            body={"text": "content to check"}
        )
    """
    # Validate URL
    if not is_valid_url(url):
        return _http_error_response(f"Invalid URL: {url}")

    # Validate and normalize method
    method = method.upper()
    allowed_methods = {"GET", "POST", "PUT", "DELETE", "PATCH"}
    if method not in allowed_methods:
        return _http_error_response(
            f"Invalid HTTP method: {method}. Allowed: {', '.join(allowed_methods)}"
        )

    # Apply timeout limits
    if timeout is None:
        timeout = _HTTP_DEFAULT_TIMEOUT
    else:
        timeout = min(max(0.1, timeout), _HTTP_MAX_TIMEOUT)

    # Get the global cached async HTTP client
    client = get_async_httpx_client(
        llm_provider=httpxSpecialProvider.GuardrailCallback,
        params={"timeout": httpx.Timeout(timeout=timeout, connect=5.0)},
    )

    try:
        response = await _execute_http_request(
            client, method, url, headers, body, timeout
        )
        return _http_success_response(response)

    except httpx.TimeoutException as e:
        verbose_proxy_logger.warning(f"Custom code http_request timeout: {e}")
        return _http_error_response(f"Request timeout after {timeout}s")
    except httpx.HTTPStatusError as e:
        # Return the response even for non-2xx status codes
        return _http_success_response(e.response)
    except httpx.RequestError as e:
        verbose_proxy_logger.warning(f"Custom code http_request error: {e}")
        return _http_error_response(f"Request failed: {str(e)}")
    except Exception as e:
        verbose_proxy_logger.warning(f"Custom code http_request unexpected error: {e}")
        return _http_error_response(f"Unexpected error: {str(e)}")


async def _execute_http_request(
    client: Any,
    method: str,
    url: str,
    headers: Optional[Dict[str, str]],
    body: Optional[Any],
    timeout: float,
) -> httpx.Response:
    """Execute the HTTP request using the appropriate client method."""
    json_body, data_body = _prepare_http_body(body)

    if method == "GET":
        return await client.get(url=url, headers=headers)
    elif method == "POST":
        return await client.post(
            url=url, headers=headers, json=json_body, data=data_body, timeout=timeout
        )
    elif method == "PUT":
        return await client.put(
            url=url, headers=headers, json=json_body, data=data_body, timeout=timeout
        )
    elif method == "DELETE":
        return await client.delete(
            url=url, headers=headers, json=json_body, data=data_body, timeout=timeout
        )
    elif method == "PATCH":
        return await client.patch(
            url=url, headers=headers, json=json_body, data=data_body, timeout=timeout
        )
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")


async def http_get(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Make an async HTTP GET request.

    Convenience wrapper around http_request for GET requests.

    Args:
        url: The URL to request
        headers: Optional dict of HTTP headers
        timeout: Optional timeout in seconds

    Returns:
        Same as http_request
    """
    return await http_request(url=url, method="GET", headers=headers, timeout=timeout)


async def http_post(
    url: str,
    body: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Make an async HTTP POST request.

    Convenience wrapper around http_request for POST requests.

    Args:
        url: The URL to request
        body: Optional request body (will be JSON-encoded if dict/list)
        headers: Optional dict of HTTP headers
        timeout: Optional timeout in seconds

    Returns:
        Same as http_request
    """
    return await http_request(
        url=url, method="POST", headers=headers, body=body, timeout=timeout
    )


# =============================================================================
# Code Detection Primitives
# =============================================================================


# Common code patterns for detection
_CODE_PATTERNS = {
    "sql": [
        r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE)\b.*\b(FROM|INTO|TABLE|SET|WHERE)\b",
        r"\b(SELECT)\s+[\w\*,\s]+\s+FROM\s+\w+",
        r"\b(INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM)\b",
    ],
    "python": [
        r"^\s*(def|class|import|from|if|for|while|try|except|with)\s+",
        r"^\s*@\w+",  # decorators
        r"\b(print|len|range|str|int|float|list|dict|set)\s*\(",
    ],
    "javascript": [
        r"\b(function|const|let|var|class|import|export)\s+",
        r"=>",  # arrow functions
        r"\b(console\.(log|error|warn))\s*\(",
    ],
    "typescript": [
        r":\s*(string|number|boolean|any|void|never)\b",
        r"\b(interface|type|enum)\s+\w+",
        r"<[A-Z]\w*>",  # generics
    ],
    "java": [
        r"\b(public|private|protected)\s+(static\s+)?(class|void|int|String)\b",
        r"\bSystem\.(out|err)\.print",
    ],
    "go": [
        r"\bfunc\s+\w+\s*\(",
        r"\b(package|import)\s+",
        r":=",  # short variable declaration
    ],
    "rust": [
        r"\b(fn|let|mut|impl|struct|enum|pub|mod)\s+",
        r"->",  # return type
        r"\b(println!|format!)\s*\(",
    ],
    "shell": [
        r"^#!.*\b(bash|sh|zsh)\b",
        r"\b(echo|grep|sed|awk|cat|ls|cd|mkdir|rm)\s+",
        r"\$\{?\w+\}?",  # variable expansion
    ],
    "html": [
        r"<\s*(html|head|body|div|span|p|a|img|script|style)\b[^>]*>",
        r"</\s*(html|head|body|div|span|p|a|script|style)\s*>",
    ],
    "css": [
        r"\{[^}]*:\s*[^}]+;[^}]*\}",
        r"@(media|keyframes|import|font-face)\b",
    ],
}


def detect_code(text: str) -> bool:
    """
    Check if text contains code of any language.

    Args:
        text: The text to check

    Returns:
        True if code is detected, False otherwise
    """
    return len(detect_code_languages(text)) > 0


def detect_code_languages(text: str) -> List[str]:
    """
    Detect which programming languages are present in text.

    Args:
        text: The text to analyze

    Returns:
        List of detected language names
    """
    detected = []
    for lang, patterns in _CODE_PATTERNS.items():
        for pattern in patterns:
            try:
                if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
                    detected.append(lang)
                    break  # Only add each language once
            except re.error:
                continue
    return detected


def contains_code_language(text: str, languages: List[str]) -> bool:
    """
    Check if text contains code from specific languages.

    Args:
        text: The text to check
        languages: List of language names to check for

    Returns:
        True if any of the specified languages are detected
    """
    detected = detect_code_languages(text)
    return any(lang.lower() in [d.lower() for d in detected] for lang in languages)


# =============================================================================
# Text Utility Primitives
# =============================================================================


def contains(text: str, substring: str) -> bool:
    """
    Check if text contains a substring.

    Args:
        text: The text to search in
        substring: The substring to find

    Returns:
        True if substring is found, False otherwise
    """
    return substring in text


def contains_any(text: str, substrings: List[str]) -> bool:
    """
    Check if text contains any of the given substrings.

    Args:
        text: The text to search in
        substrings: List of substrings to find

    Returns:
        True if any substring is found, False otherwise
    """
    return any(s in text for s in substrings)


def contains_all(text: str, substrings: List[str]) -> bool:
    """
    Check if text contains all of the given substrings.

    Args:
        text: The text to search in
        substrings: List of substrings to find

    Returns:
        True if all substrings are found, False otherwise
    """
    return all(s in text for s in substrings)


def word_count(text: str) -> int:
    """
    Count the number of words in text.

    Args:
        text: The text to count words in

    Returns:
        Number of words
    """
    return len(text.split())


def char_count(text: str) -> int:
    """
    Count the number of characters in text.

    Args:
        text: The text to count characters in

    Returns:
        Number of characters
    """
    return len(text)


def lower(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def upper(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


def trim(text: str) -> str:
    """Remove leading and trailing whitespace."""
    return text.strip()


# =============================================================================
# Primitives Registry
# =============================================================================


def get_custom_code_primitives() -> Dict[str, Any]:
    """
    Get all primitives to inject into the custom code environment.

    Returns:
        Dict of function name to function
    """
    return {
        # Result types
        "allow": allow,
        "block": block,
        "modify": modify,
        # Regex
        "regex_match": regex_match,
        "regex_match_all": regex_match_all,
        "regex_replace": regex_replace,
        "regex_find_all": regex_find_all,
        # JSON
        "json_parse": json_parse,
        "json_stringify": json_stringify,
        "json_schema_valid": json_schema_valid,
        # URL
        "extract_urls": extract_urls,
        "is_valid_url": is_valid_url,
        "all_urls_valid": all_urls_valid,
        "get_url_domain": get_url_domain,
        # HTTP (async)
        "http_request": http_request,
        "http_get": http_get,
        "http_post": http_post,
        # Code detection
        "detect_code": detect_code,
        "detect_code_languages": detect_code_languages,
        "contains_code_language": contains_code_language,
        # Text utilities
        "contains": contains,
        "contains_any": contains_any,
        "contains_all": contains_all,
        "word_count": word_count,
        "char_count": char_count,
        "lower": lower,
        "upper": upper,
        "trim": trim,
        # Python builtins (safe subset)
        "len": len,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "True": True,
        "False": False,
        "None": None,
    }
