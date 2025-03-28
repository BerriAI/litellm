# Testing for `litellm/` 

This directory 1:1 maps the the `litellm/` directory, and can only contain mocked tests. 

The point of this is to:
1. Increase test coverage of `litellm/`
2. Make it easy for contributors to add tests for the `litellm/` package and easily run tests without needing LLM API keys. 


## File name conventions

- `litellm/proxy/test_caching_routes.py` maps to `litellm/proxy/caching_routes.py`
- `test_<filename>.py` maps to `litellm/<filename>.py`











