import base64
import json
import types

from litellm.litellm_core_utils.gateway_io_logging import (
    attach_gateway_io_to_spend_log_payload,
    build_gateway_io_metadata,
    capture_external_egress,
)


def test_gateway_io_logging_default_deny_and_credential_substitution() -> None:
    lp = {
        "litellm_credential_name": "OpenAI-default",
        "litellm_metadata": {
            "_gateway_credential_name": "OpenAI-develop-dx_vd-20261022",
            "_gateway_credential_source": "model_config",
        },
        "proxy_server_request": {
            "url": "/v1/responses?api-version=2025-01-01&p1=fragA",
            "headers": {
                "authorization": "Bearer sk-virtual-raw-1234567890",
                "content-type": "application/json",
                "x-custom-part": "fragA",
            },
            "body": {
                "model": "openai/gpt-6.1-terra",
                "input": "hello world",
            },
        },
    }
    meta = {
        "user_api_key_alias": "vk-develop-dx_vd-test",
        "status": "success",
    }
    cap = capture_external_egress(
        api_key="sk-real-secret-1234567890",
        additional_args={
            "api_base": "https://api.openai.com/v1/responses?api-version=2025-01-01&key=raw-q-secret",
            "headers": {
                "authorization": "Bearer sk-real-secret-1234567890",
                "content-type": "application/json",
                "x-stainless-os": "Linux",
                "x-part-1": "fragB",
            },
            "complete_input_dict": {
                "model": "gpt-6.1-terra",
                "input": "hello world",
                "instructions": "TOP_SECRET_SYS_PROMPT",
                "max_output_tokens": 1024,
                "access_programs": {"cyber": "daybreak_blue"},
                "chunk_1": "secPart1",
            },
        },
        model_call_details={"litellm_params": lp},
    )
    res = build_gateway_io_metadata(
        kwargs={
            "_gateway_external_egress": cap,
            "response_headers": {
                "llm_provider-x-request-id": "req_123",
                "llm_provider-x-ratelimit-remaining-tokens": "99999",
                "llm_provider-set-cookie": "sid=secret",
            },
        },
        metadata=meta,
        litellm_params=lp,
        response_obj_dict={
            "model": "gpt-6.1-terra",
            "access_programs": {"cyber": "daybreak_blue"},
        },
        standard_logging_payload=None,
        custom_llm_provider="openai",
        api_base="https://api.openai.com/v1/responses",
    )

    serialized = json.dumps(res)
    for leaked in (
        "sk-real-secret",
        "sk-virtual-raw",
        "sid=secret",
        "raw-q-secret",
        "fragA",
        "fragB",
        "secPart1",
        "TOP_SECRET_SYS_PROMPT",
    ):
        assert leaked not in serialized

    assert "api-version=2025-01-01" in res["internal"]["ingress"]["route"]
    assert res["internal"]["ingress"]["headers"]["authorization"] == "Bearer <credential:vk-develop-dx_vd-test>"
    assert res["internal"]["ingress"]["headers"]["x-custom-part"] == "***"
    assert res["internal"]["ingress"]["litellm_process"]["header"]["authorization"] == "replace:key_alias"
    assert res["internal"]["ingress"]["litellm_process"]["header"]["x-custom-part"] == "mask"

    ext_egress = res["external"]["egress"]
    assert ext_egress["credential_name"] == "OpenAI-develop-dx_vd-20261022"
    assert ext_egress["headers"]["authorization"] == "Bearer <credential:OpenAI-develop-dx_vd-20261022>"
    assert ext_egress["headers"]["x-stainless-os"] == "Linux"
    assert ext_egress["headers"]["x-part-1"] == "***"
    assert ext_egress["body_params"]["max_output_tokens"] == 1024
    assert "instructions_bytes" in ext_egress["body_params"]
    assert "chunk_1_bytes" in ext_egress["body_params"]
    assert ext_egress["litellm_process"]["header"]["authorization"] == "replace:credential_name"
    assert ext_egress["litellm_process"]["header"]["x-stainless-os"] == "add"
    assert ext_egress["litellm_process"]["header"]["x-part-1"] == "mask"
    assert ext_egress["litellm_process"]["body"]["access_programs"] == "add"
    assert ext_egress["litellm_process"]["body"]["model"] == "replace:model"

    ext_ingress = res["external"]["ingress"]
    assert ext_ingress["headers"]["x-request-id"] == "req_123"
    assert ext_ingress["headers"]["x-ratelimit-remaining-tokens"] == "99999"
    assert ext_ingress["headers"]["set-cookie"] == "***"
    assert ext_ingress["litellm_process"]["header"]["set-cookie"] == "mask"


def test_gateway_io_logging_blocks_encoded_and_truncated_pem_secrets() -> None:
    raw_sec = "myRealSecretKey9988"
    b64_sec = base64.b64encode(raw_sec.encode("utf-8")).decode("ascii")
    pem_trunc = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSj"
    lp = {
        "api_key": raw_sec,
        "vertex_credentials": {"private_key": pem_trunc, "private_key_id": "pkid-secret-12345"},
        "litellm_metadata": {"_gateway_credential_name": "Vertex-prod"},
    }
    res = build_gateway_io_metadata(
        kwargs={"api_key": raw_sec, "exception": types.SimpleNamespace(status_code=503)},
        metadata={
            "error_information": {
                "error_message": (
                    f"Err b64={b64_sec} pem={pem_trunc} "
                    "req_abc12345678901234567890123456 rand=9f8e7d6c5b4a3210fedcba9876543210"
                )
            }
        },
        litellm_params=lp,
        response_obj_dict={},
        standard_logging_payload=None,
        custom_llm_provider="vertex_ai",
        api_base="https://u:pw@api.openai.com/v1/responses",
        response_obj=None,
    )
    out = json.dumps(res)
    for leaked in (raw_sec, b64_sec, "MIIEvQIB", "pkid-secret-12345", "9f8e7d6c5b4a3210", "pw"):
        assert leaked not in out
    assert res["external"]["ingress"]["http_status"] == 503
    assert "req_abc12345678901234567890123456" in res["external"]["ingress"]["error_message"]


def test_attach_gateway_io_to_spend_log_payload() -> None:
    payload = {
        "custom_llm_provider": "openai",
        "api_base": "https://api.openai.com/v1/responses",
        "metadata": json.dumps({"status": "success", "_gateway_credential_name": "OpenAI-prod"}),
    }
    updated = attach_gateway_io_to_spend_log_payload(
        payload,
        kwargs={"litellm_params": {"litellm_metadata": {"status": "success"}}},
        response_obj={"model": "gpt-6.1-terra"},
    )
    parsed_meta = json.loads(updated["metadata"])
    assert "gateway_io" in parsed_meta
    assert "_gateway_credential_name" not in parsed_meta
    assert "internal" in parsed_meta["gateway_io"]
    assert "external" in parsed_meta["gateway_io"]



def test_gateway_io_logging_redacts_rejected_model_schema_desc_and_ip() -> None:
    cap = capture_external_egress(
        additional_args={
            "api_base": "https://api.openai.com/v1/responses",
            "headers": {"x-correlation-id": "client-203.0.113.9"},
            "complete_input_dict": {
                "model": "opus-4.6 Please summarize\nPatient has diabetes",
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "med_record",
                        "strict": True,
                        "description": "PRIVATE_SCHEMA_DESC",
                        "schema": {"type": "object"},
                    },
                },
            },
        },
        model_call_details={"litellm_params": {}},
    )
    res = build_gateway_io_metadata(
        kwargs={"_gateway_external_egress": cap, "exception": types.SimpleNamespace(status_code=400)},
        metadata={"status": "failure"},
        litellm_params={
            "proxy_server_request": {
                "url": "/v1/responses",
                "headers": {"x-correlation-id": "client-203.0.113.9"},
                "body": {"model": "opus-4.6 Please summarize\nPatient has diabetes"},
            }
        },
        response_obj_dict={},
        standard_logging_payload=None,
        custom_llm_provider="openai",
        api_base="",
        response_obj=None,
        sanitized_error_info={"error_class": "BadRequestError", "error_message": "bad model"},
        sanitized_model="invalid-model",
        sanitized_model_group="invalid-model",
        ip_redactor=lambda text: text.replace("203.0.113.9", "***"),
    )
    out = json.dumps(res)
    for leaked in ("Patient has diabetes", "PRIVATE_SCHEMA_DESC", "203.0.113.9"):
        assert leaked not in out
    assert res["internal"]["ingress"]["model"] == "invalid-model"
    assert res["external"]["egress"]["body_params"]["model"] == "invalid-model"
    assert res["external"]["egress"]["body_params"]["response_format"]["type"] == "json_schema"
    assert res["external"]["egress"]["body_params"]["response_format"]["strict"] is True
    assert res["external"]["egress"]["headers"]["x-correlation-id"] == "client-***"



def test_gateway_io_logging_redacts_secrets_in_response_access_programs() -> None:
    raw_sec = "myProviderSecretKey9988"
    res = build_gateway_io_metadata(
        kwargs={"api_key": raw_sec},
        metadata={"status": "success"},
        litellm_params={"api_key": raw_sec, "litellm_metadata": {"_gateway_credential_name": "OpenAI-prod"}},
        response_obj_dict={"model": "gpt-6.1-terra", "access_programs": {"cyber": "daybreak_blue", "echo": raw_sec}},
        standard_logging_payload=None,
        custom_llm_provider="openai",
        api_base="https://api.openai.com/v1/responses",
        response_obj=None,
    )
    out = json.dumps(res)
    assert raw_sec not in out
    assert res["external"]["ingress"]["access_programs"]["cyber"] == "daybreak_blue"
    assert res["external"]["ingress"]["access_programs"]["echo"] == "<credential:OpenAI-prod>"
