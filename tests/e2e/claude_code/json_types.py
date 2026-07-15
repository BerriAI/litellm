"""Shared JSON typing vocabulary for the Claude Code compat suite.

Everything the suite parses off a wire or a file (stream-json events from
the `claude` CLI, npm packuments, manifest YAML, results artifacts) is
JSON-shaped. `JSONValue` models that shape recursively so strict type
checking can narrow payloads with plain `isinstance` checks, and the
`TypeAdapter`s validate untyped input (`json.loads`, `yaml.safe_load`,
HTTP bodies) into the typed shape at the boundary instead of letting
`Any` leak through the suite.
"""

from pydantic import TypeAdapter

type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
type JSONObject = dict[str, JSONValue]

JSON_VALUE_ADAPTER = TypeAdapter[JSONValue](JSONValue)
JSON_OBJECT_ADAPTER = TypeAdapter[JSONObject](JSONObject)
