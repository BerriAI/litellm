# starlette

Starlette's route template syntax, for route templates Python LiteLLM registers with FastAPI and matches with starlette, such as `LiteLLMRoutes` groups and pass-through `path`s. A crate that reads one of those templates depends on this crate and never parses `{name:convertor}` itself

- `RouteTemplate::parse` follows `starlette.routing.compile_path`: `PARAM_REGEX` names, the five `CONVERTOR_TYPES`, and the port a host template drops. It never fails, where starlette raises on an unknown convertor, because starlette refuses such a route at registration, so no live route has one. It reads as `str`
- `RouteTemplate::matches` is the regex `compile_path` builds. Other matching rules LiteLLM writes over the same templates, such as `RouteChecks._route_matches_pattern`, belong to their callers, which pass what each parameter takes to `matches_with` rather than walking `parts()` themselves, so every matcher stays polynomial however many parameters a template has
- Every case in `tests/` takes its expected value from running `bool(compile_path(template)[0].match(route))` on starlette, never from reasoning. Run it before adding or changing a case, and update the version the tests cite when the proxy's starlette changes a convertor
