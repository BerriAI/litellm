import os
from litellm import completion

oci_region = os.environ.get("OCI_REGION")
oci_user = os.environ.get("OCI_USER")
oci_fingerprint = os.environ.get("OCI_FINGERPRINT")
oci_tenancy = os.environ.get("OCI_TENANCY")
oci_key = os.environ.get("OCI_KEY")
oci_compartment_id = os.environ.get("OCI_COMPARTMENT_ID")

response = completion(
    # model="oci/meta.llama-3.3-70b-instruct",
    model="oci/xai.grok-4",
    # model="oci/cohere.command-latest",
    # model="oci/cohere.command-r-plus-08-2024",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        
        {"role": "user", "content": "What is the capital of Brazil?"},
        
        # Modelo chama a ferramenta
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tool_call_1",
                    "type": "function",
                    "function": {
                        "name": "get_capital",
                        "arguments": '{"country": "Brazil"}'
                    }
                }
            ]
        },

        # Sua aplicação responde a tool_call
        {
            "role": "tool",
            "tool_call_id": "tool_call_1",
            "content": '{"capital": "Brasília"}'
        },

        # Modelo responde com base na resposta da ferramenta
        {
            "role": "assistant",
            "content": "The capital of Brazil is Brasília."
        },

        # Novo input do usuário
        {"role": "user", "content": "What about France?"},

        # Modelo chama a ferramenta
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "tool_call_1",
                    "type": "function",
                    "function": {
                        "name": "get_capital",
                        "arguments": '{"country": "France"}'
                    }
                }
            ]
        },

        # Sua aplicação responde a tool_call
        {
            "role": "tool",
            "tool_call_id": "tool_call_1",
            "content": '{"capital": "Paris"}'
        },
    ],
    oci_region=oci_region,
    oci_user=oci_user,
    oci_fingerprint=oci_fingerprint,
    oci_tenancy=oci_tenancy,
    oci_key=oci_key,
    oci_compartment_id=oci_compartment_id,
    stream=True,
    tools=[{
        "type": "function",
        "function": {
            "name": "get_capital",
            "description": "Get the capital of a country.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "The name of the country."
                    }
                },
                "required": ["country"]
            }
        }
    }]
)

for chunk in response:
    print(chunk)

# print(response)
