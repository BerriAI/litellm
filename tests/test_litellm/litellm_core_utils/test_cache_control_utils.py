from litellm.litellm_core_utils.cache_control_utils import parse_cache_control_header


def test_parse_cache_control_header_none():
    assert parse_cache_control_header(None) == {}


def test_parse_cache_control_header_empty():
    assert parse_cache_control_header("") == {}


def test_parse_cache_control_header_no_cache():
    res = parse_cache_control_header("no-cache")
    assert res == {"no-cache": True}


def test_parse_cache_control_header_no_store():
    res = parse_cache_control_header("no-store")
    assert res == {"no-store": True}


def test_parse_cache_control_header_max_age():
    res = parse_cache_control_header("max-age=300")
    assert res.get("s-maxage") == 300
    assert res.get("s-max-age") == 300
    assert res.get("ttl") == 300


def test_parse_cache_control_header_s_maxage():
    res = parse_cache_control_header("s-maxage=600")
    assert res.get("s-maxage") == 600
    assert res.get("ttl") == 600


def test_parse_cache_control_header_multiple_directives():
    res = parse_cache_control_header("no-cache, max-age=120")
    assert res.get("no-cache") is True
    assert res.get("s-maxage") == 120
    assert res.get("ttl") == 120


def test_parse_cache_control_header_unspaced_comma():
    res = parse_cache_control_header("no-cache,max-age=120")
    assert res.get("no-cache") is True
    assert res.get("s-maxage") == 120


def test_parse_cache_control_header_case_and_whitespace():
    res = parse_cache_control_header("  NO-CACHE , MAX-AGE=300  ")
    assert res.get("no-cache") is True
    assert res.get("s-maxage") == 300


def test_parse_cache_control_header_quoted_value():
    res = parse_cache_control_header('max-age="300"')
    assert res.get("s-maxage") == 300


def test_parse_cache_control_header_custom_directive():
    res = parse_cache_control_header("custom-directive=hello")
    assert res.get("custom-directive") == "hello"
