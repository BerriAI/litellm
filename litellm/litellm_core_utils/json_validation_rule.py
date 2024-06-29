import json


def validate_schema(schema: dict, response: str):
    """
    Validate if the returned json response follows the schema.

    Params:
    - schema - dict: JSON schema
    - response - str: Received json response as string.
    """
    from jsonschema import ValidationError, validate

    from litellm import JSONSchemaValidationError

    response_dict = json.loads(response)

    try:
        validate(response_dict, schema=schema)
    except ValidationError:
        raise JSONSchemaValidationError(
            model="", llm_provider="", raw_response=response, schema=json.dumps(schema)
        )
