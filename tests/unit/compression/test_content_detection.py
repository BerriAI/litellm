"""Code detection: the keyword list must see every construct it names."""

from litellm.compression.content_detection import detect_content_type


def test_include_and_require_are_code_keywords():
    """`#include` and `require(` can never match a pattern that wants a word char after them."""
    source = '#include <stdio.h>\nstatic int add(int a, int b) {\n    require("math");\n}\n'

    assert detect_content_type(source) == "code"


def test_repeated_require_calls_are_code():
    source = "require('fs')\nrequire('path')\nconst os = require('os')\n"

    assert detect_content_type(source) == "code"


def test_prose_is_still_text():
    assert detect_content_type("The meeting is tomorrow at noon.\nPlease bring the report.\n") == "text"


def test_json_is_still_json():
    assert detect_content_type('{"candidates": [{"text": "hello"}]}') == "json"
