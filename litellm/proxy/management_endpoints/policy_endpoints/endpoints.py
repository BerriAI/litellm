"""
POLICY MANAGEMENT

All /policy management endpoints

/policy/validate - Validate a policy configuration
/policy/list - List all loaded policies
/policy/info - Get information about a specific policy
/policy/templates - Get policy templates (GitHub with local fallback)
"""

import copy
import json
import os
from typing import (TYPE_CHECKING, Any, AsyncIterator, List, Literal, Optional,
                    cast)

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.constants import (COMPETITOR_LLM_TEMPERATURE,
                               DEFAULT_COMPETITOR_DISCOVERY_MODEL,
                               MAX_COMPETITOR_NAMES)
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.openai.chat.guardrail_translation.handler import \
    OpenAIChatCompletionsHandler
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.guardrails.guardrail_hooks.custom_code import (
    RESPONSE_REJECTION_GUARDRAIL_CODE, CustomCodeGuardrail)
from litellm.proxy.guardrails.guardrail_registry import GuardrailRegistry
from litellm.proxy.management_helpers.utils import management_endpoint_wrapper
from litellm.proxy.policy_engine.policy_registry import get_policy_registry
from litellm.proxy.policy_engine.policy_resolver import PolicyResolver
from litellm.types.proxy.policy_engine import (PolicyGuardrailsResponse,
                                               PolicyInfoResponse,
                                               PolicyListResponse,
                                               PolicyMatchContext,
                                               PolicyScopeResponse,
                                               PolicySummaryItem,
                                               PolicyTestResponse,
                                               PolicyValidateRequest,
                                               PolicyValidationResponse)
from litellm.types.utils import GenericGuardrailAPIInputs, ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import \
        Logging as LiteLLMLoggingObj

router = APIRouter()


class GuardrailApplyError(Exception):
    """
    Raised when a guardrail's apply_guardrail fails during apply_policies.

    Consumers (e.g. Compliance UI) can use guardrail_name and message to show
    which guardrail triggered and the error reason.
    """

    def __init__(self, guardrail_name: str, message: str) -> None:
        self.guardrail_name = guardrail_name
        self.message = message
        super().__init__(f"Guardrail '{guardrail_name}' failed: {message}")


class GuardrailErrorEntry(TypedDict):
    """One guardrail failure for ApplyPoliciesResult.guardrail_errors."""

    guardrail_name: str
    message: str


class _ApplyPoliciesResultBase(TypedDict):
    """Base result of apply_policies: inputs plus any guardrail failures."""

    inputs: GenericGuardrailAPIInputs
    guardrail_errors: List[GuardrailErrorEntry]


class ApplyPoliciesResult(_ApplyPoliciesResultBase, total=False):
    """Result of apply_policies. agent_response set when agent_id provided."""

    agent_response: Any


class _ApplyPoliciesPerItemResultBase(TypedDict):
    """Base result for one input when using inputs_list."""

    inputs: GenericGuardrailAPIInputs
    guardrail_errors: List[GuardrailErrorEntry]


class ApplyPoliciesPerItemResult(_ApplyPoliciesPerItemResultBase, total=False):
    """Result for one input when using inputs_list. agent_response set when agent_id provided."""

    agent_response: Any


class ApplyPoliciesListResult(TypedDict):
    """Result when using inputs_list: one result per input."""

    results: List[ApplyPoliciesPerItemResult]


async def apply_policies(
    policy_names: Optional[list[str]],
    inputs: GenericGuardrailAPIInputs,
    request_data: dict,
    input_type: Literal["request", "response"],
    proxy_logging_obj: "LiteLLMLoggingObj",
    guardrail_names: Optional[list[str]] = None,
) -> ApplyPoliciesResult:
    """
    Apply guardrails to inputs from policy names and/or a direct list of guardrail names.

    Runs all guardrails in order; if one fails, the error is recorded and execution
    continues so that all inputs can complete testing and all guardrail failures are
    collected. No exception is raised; failures are returned in guardrail_errors.

    Guardrails can be specified in two ways (both can be used together; names are merged):
    - policy_names: resolve guardrails from the policy registry (with inheritance).
    - guardrail_names: use this list of guardrail names directly (no policy registry needed).

    Returns:
        ApplyPoliciesResult with "inputs" (final GenericGuardrailAPIInputs) and
        "guardrail_errors" (list of {"guardrail_name", "message"} for each failure).
    """
    guardrail_errors: List[GuardrailErrorEntry] = []

    guardrail_name_set: set[str] = set()

    if guardrail_names:
        guardrail_name_set.update(guardrail_names)

    if policy_names:
        registry = get_policy_registry()
        if not registry.is_initialized():
            verbose_proxy_logger.debug(
                "apply_policies: policy engine not initialized, skipping policy-resolved guardrails"
            )
        else:
            policies = registry.get_all_policies()
            for policy_name in policy_names:
                resolved = PolicyResolver.resolve_policy_guardrails(
                    policy_name=policy_name,
                    policies=policies,
                    context=None,
                )
                guardrail_name_set.update(resolved.guardrails)

    if not guardrail_name_set:
        return {"inputs": inputs, "guardrail_errors": guardrail_errors}

    guardrail_registry = GuardrailRegistry()
    current_inputs = cast(GenericGuardrailAPIInputs, dict(inputs))

    for guardrail_name in sorted(guardrail_name_set):
        callback = guardrail_registry.get_initialized_guardrail_callback(
            guardrail_name=guardrail_name
        )
        if callback is None:
            verbose_proxy_logger.debug(
                "apply_policies: guardrail '%s' not found, skipping",
                guardrail_name,
            )
            continue
        if not isinstance(callback, CustomGuardrail):
            continue
        if "apply_guardrail" not in type(callback).__dict__:
            verbose_proxy_logger.debug(
                "apply_policies: guardrail '%s' has no apply_guardrail, skipping",
                guardrail_name,
            )
            continue

        try:
            current_inputs = await callback.apply_guardrail(
                inputs=current_inputs,
                request_data=request_data,
                input_type=input_type,
                logging_obj=proxy_logging_obj,
            )
        except Exception as e:
            error_reason = str(e)
            verbose_proxy_logger.debug(
                "apply_policies: guardrail '%s' failed: %s",
                guardrail_name,
                error_reason,
            )
            guardrail_errors.append(
                GuardrailErrorEntry(
                    guardrail_name=guardrail_name,
                    message=error_reason,
                )
            )
            # Continue to next guardrail; current_inputs unchanged for this failure

    return {"inputs": current_inputs, "guardrail_errors": guardrail_errors}


def _chat_body_from_inputs(
    inputs: GenericGuardrailAPIInputs, agent_id: str, request_data: dict
) -> dict:
    """Build a chat completion request body from guardrail inputs and agent_id."""
    messages: List[dict]
    structured = inputs.get("structured_messages")
    texts = inputs.get("texts")
    if structured:
        messages = list(structured)  # type: ignore[arg-type]
    elif texts:
        if len(texts) == 1:
            messages = [{"role": "user", "content": texts[0]}]
        else:
            messages = [{"role": "user", "content": "\n".join(texts)}]
    else:
        messages = [{"role": "user", "content": "Hello"}]
    body: dict = {"model": agent_id, "messages": messages, "stream": False}
    if request_data:
        body.setdefault("metadata", {}).update(request_data)
    return body


def _request_with_json_body(body: dict) -> Request:
    """Create a Starlette Request that will return the given dict as parsed JSON body."""
    body_bytes = json.dumps(body).encode()
    received: List[bool] = [False]

    async def receive() -> dict:
        if received[0]:
            return {"type": "http.disconnect"}
        received[0] = True
        return {"type": "http.request", "body": body_bytes, "more_body": False}

    scope: dict = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "scheme": "http",
        "server": ("localhost", 8000),
        "client": ("127.0.0.1", 0),
        "root_path": "",
        "app": None,
        "asgi": {"version": "3.0", "spec_version": "2.0"},
    }
    return Request(scope, receive=receive)


class TestPoliciesAndGuardrailsRequest(BaseModel):
    """Request body for POST /utils/test_policies_and_guardrails."""

    policy_names: Optional[List[str]] = Field(
        default=None, description="Policy names to resolve guardrails from"
    )
    guardrail_names: Optional[List[str]] = Field(
        default=None, description="Guardrail names to apply directly"
    )
    inputs_list: List[GenericGuardrailAPIInputs] = Field(
        default=[],
        description="List of GenericGuardrailAPIInputs; each item processed separately (for batch compliance testing).",
    )
    request_data: dict = Field(
        default_factory=dict, description="Request context (model, user_id, etc.)"
    )
    input_type: Literal["request", "response"] = Field(
        default="request", description="Whether inputs are request or response"
    )
    agent_id: Optional[str] = Field(
        default=None,
        description="When set, call chat completion with this model/agent for each input and include the response in the result.",
    )


@router.post(
    "/utils/test_policies_and_guardrails",
    tags=["utils"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def test_policies_and_guardrails(
    request: Request,
    data: TestPoliciesAndGuardrailsRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Apply policies and/or guardrails to inputs (for compliance UI testing).

    Use inputs_list for batch testing: each input is processed as a separate call so
    per-input block/allow and errors are returned.

    Use inputs for a single call (legacy).
    """
    from litellm.litellm_core_utils.litellm_logging import \
        Logging as LiteLLMLoggingObj
    from litellm.proxy.proxy_server import chat_completion, proxy_logging_obj
    from litellm.proxy.utils import handle_exception_on_proxy

    def _serialize_chat_response(response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(exclude_unset=True)
        if isinstance(response, dict):
            return response
        return response

    async def _get_agent_response(
        inputs: GenericGuardrailAPIInputs,
        agent_id: str,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> Any:
        body = _chat_body_from_inputs(inputs, agent_id, data.request_data)
        req = _request_with_json_body(body)
        resp = Response()
        result = await chat_completion(
            request=req,
            fastapi_response=resp,
            model=agent_id,
            user_api_key_dict=user_api_key_dict,
        )
        return _serialize_chat_response(result)

    try:
        logging_obj = cast(LiteLLMLoggingObj, proxy_logging_obj)

        results: List[ApplyPoliciesPerItemResult] = []
        for inp in data.inputs_list:
            item_result = await apply_policies(
                policy_names=data.policy_names,
                inputs=inp,
                request_data=data.request_data,
                input_type=data.input_type,
                proxy_logging_obj=logging_obj,
                guardrail_names=data.guardrail_names,
            )
            item: ApplyPoliciesPerItemResult = {
                "inputs": item_result["inputs"],
                "guardrail_errors": item_result["guardrail_errors"],
            }
            if data.agent_id is not None:
                item["agent_response"] = await _get_agent_response(
                    item_result["inputs"],
                    data.agent_id,
                    user_api_key_dict,
                )
                # run response through response_rejection_guardrail (reuses handler extraction + apply)
                response_rejection_guardrail = CustomCodeGuardrail(
                    custom_code=RESPONSE_REJECTION_GUARDRAIL_CODE,
                    guardrail_name="response_rejection",
                )
                try:
                    model_response = ModelResponse.model_validate(
                        item["agent_response"]
                    )
                    handler = OpenAIChatCompletionsHandler()
                    await handler.process_output_response(
                        response=model_response,
                        guardrail_to_apply=response_rejection_guardrail,
                        litellm_logging_obj=logging_obj,
                        user_api_key_dict=user_api_key_dict,
                    )
                except Exception as guardrail_err:
                    item["guardrail_errors"] = list(item["guardrail_errors"])
                    detail = getattr(guardrail_err, "detail", None)
                    if isinstance(detail, dict) and "error" in detail:
                        err_msg = detail["error"]
                    else:
                        err_msg = str(detail if detail is not None else guardrail_err)
                    item["guardrail_errors"].append(
                        GuardrailErrorEntry(
                            guardrail_name="response_rejection",
                            message=err_msg,
                        )
                    )
            results.append(item)
        return ApplyPoliciesListResult(results=results)
        raise ValueError("Either inputs or inputs_list must be provided")
    except Exception as e:
        raise handle_exception_on_proxy(e)


@router.post(
    "/policy/validate",
    tags=["policy management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyValidationResponse,
)
@management_endpoint_wrapper
async def validate_policy(
    request: Request,
    data: PolicyValidateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PolicyValidationResponse:
    """
    Validate a policy configuration before applying it.

    Checks:
    - All referenced guardrails exist in the guardrail registry
    - All non-wildcard team aliases exist in the database
    - All non-wildcard key aliases exist in the database
    - Inheritance chains are valid (no cycles, parents exist)
    - Scope patterns are syntactically valid

    Returns:
    - valid: True if the policy configuration is valid (no blocking errors)
    - errors: List of blocking validation errors
    - warnings: List of non-blocking validation warnings

    Example request:
    ```json
    {
        "policies": {
            "global-baseline": {
                "guardrails": {
                    "add": ["pii_blocker", "phi_blocker"]
                },
                "scope": {
                    "teams": ["*"],
                    "keys": ["*"],
                    "models": ["*"]
                }
            },
            "healthcare-compliance": {
                "inherit": "global-baseline",
                "guardrails": {
                    "add": ["hipaa_audit"]
                },
                "scope": {
                    "teams": ["healthcare-team"]
                }
            }
        }
    }
    ```
    """
    from litellm.proxy.policy_engine.policy_validator import PolicyValidator
    from litellm.proxy.proxy_server import prisma_client

    verbose_proxy_logger.debug(
        f"Validating policy configuration with {len(data.policies)} policies"
    )

    validator = PolicyValidator(prisma_client=prisma_client)

    result = await validator.validate_policy_config(
        data.policies,
        validate_db=prisma_client is not None,
    )

    return result


@router.get(
    "/policy/list",
    tags=["policy management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyListResponse,
)
@management_endpoint_wrapper
async def list_policies(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PolicyListResponse:
    """
    List all loaded policies with their resolved guardrails.

    Returns information about each policy including:
    - Inheritance configuration
    - Scope (teams, keys, models)
    - Guardrails to add/remove
    - Resolved guardrails (after inheritance)
    - Inheritance chain
    """
    from litellm.proxy.policy_engine.init_policies import get_policies_summary

    summary = get_policies_summary()
    return PolicyListResponse(
        policies={
            name: PolicySummaryItem(
                inherit=data.get("inherit"),
                scope=PolicyScopeResponse(**data.get("scope", {})),
                guardrails=PolicyGuardrailsResponse(**data.get("guardrails", {})),
                resolved_guardrails=data.get("resolved_guardrails", []),
                inheritance_chain=data.get("inheritance_chain", []),
            )
            for name, data in summary.get("policies", {}).items()
        },
        total_count=summary.get("total_count", 0),
    )


@router.get(
    "/policy/info/{policy_name}",
    tags=["policy management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyInfoResponse,
)
@management_endpoint_wrapper
async def get_policy_info(
    request: Request,
    policy_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PolicyInfoResponse:
    """
    Get detailed information about a specific policy.

    Returns:
    - Policy configuration
    - Resolved guardrails (after inheritance)
    - Inheritance chain
    """
    from litellm.proxy.policy_engine.policy_registry import get_policy_registry
    from litellm.proxy.policy_engine.policy_resolver import PolicyResolver

    registry = get_policy_registry()

    if not registry.is_initialized():
        raise HTTPException(
            status_code=404,
            detail="Policy engine not initialized. No policies loaded.",
        )

    policy = registry.get_policy(policy_name)
    if policy is None:
        raise HTTPException(
            status_code=404,
            detail=f"Policy '{policy_name}' not found",
        )

    resolved = PolicyResolver.resolve_policy_guardrails(
        policy_name=policy_name, policies=registry.get_all_policies()
    )

    return PolicyInfoResponse(
        policy_name=policy_name,
        inherit=policy.inherit,
        scope=PolicyScopeResponse(
            teams=[],
            keys=[],
            models=[],
        ),
        guardrails=PolicyGuardrailsResponse(
            add=policy.guardrails.get_add(),
            remove=policy.guardrails.get_remove(),
        ),
        resolved_guardrails=resolved.guardrails,
        inheritance_chain=resolved.inheritance_chain,
    )


@router.post(
    "/policy/test",
    tags=["policy management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=PolicyTestResponse,
)
@management_endpoint_wrapper
async def test_policy_matching(
    request: Request,
    context: PolicyMatchContext,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> PolicyTestResponse:
    """
    Test which policies would match a given request context.

    This is useful for debugging and understanding policy behavior.

    Request body:
    ```json
    {
        "team_alias": "healthcare-team",
        "key_alias": "my-api-key",
        "model": "gpt-4"
    }
    ```

    Returns:
    - matching_policies: List of policy names that match
    - resolved_guardrails: Final list of guardrails that would be applied
    """
    from litellm.proxy.policy_engine.policy_matcher import PolicyMatcher
    from litellm.proxy.policy_engine.policy_registry import get_policy_registry
    from litellm.proxy.policy_engine.policy_resolver import PolicyResolver

    registry = get_policy_registry()

    if not registry.is_initialized():
        return PolicyTestResponse(
            context=context,
            matching_policies=[],
            resolved_guardrails=[],
            message="Policy engine not initialized. No policies loaded.",
        )

    policies = registry.get_all_policies()

    # Get matching policies
    matching_policy_names = PolicyMatcher.get_matching_policies(context=context)

    # Resolve guardrails
    resolved_guardrails = PolicyResolver.resolve_guardrails_for_context(
        context=context, policies=policies
    )

    return PolicyTestResponse(
        context=context,
        matching_policies=matching_policy_names,
        resolved_guardrails=resolved_guardrails,
    )


POLICY_TEMPLATES_GITHUB_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/policy_templates.json"
)


def _load_policy_templates_from_local_backup() -> list:
    """Load policy templates from local backup file (litellm/policy_templates_backup.json)."""
    backup_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "policy_templates_backup.json",
    )
    path = os.path.abspath(backup_path)
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


@router.get(
    "/policy/templates",
    tags=["policy management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def get_policy_templates(
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> list:
    """
    Get policy templates for the UI (pre-configured guardrail combinations).

    Fetches from GitHub with automatic fallback to local backup on failure.
    Set LITELLM_LOCAL_POLICY_TEMPLATES=true to skip GitHub and use local backup only.
    """
    use_local = os.getenv("LITELLM_LOCAL_POLICY_TEMPLATES", "").strip().lower() in (
        "true",
        "1",
        "yes",
    )
    if use_local:
        return _load_policy_templates_from_local_backup()

    try:
        from litellm.llms.custom_httpx.http_handler import \
            get_async_httpx_client
        from litellm.types.llms.custom_http import httpxSpecialProvider

        async_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.UI,
            params={"timeout": 10.0},
        )
        response = await async_client.get(POLICY_TEMPLATES_GITHUB_URL)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        verbose_proxy_logger.debug(
            "Failed to fetch policy templates from GitHub, using local backup: %s", e
        )

    return _load_policy_templates_from_local_backup()


class EnrichTemplateRequest(BaseModel):
    template_id: str
    parameters: dict
    model: Optional[str] = None
    competitors: Optional[List[str]] = Field(
        default=None,
        max_length=MAX_COMPETITOR_NAMES,
        description="Optional list of competitor names",
    )
    instruction: Optional[str] = Field(
        default=None,
        description="Refinement instruction for modifying the competitor list (e.g. 'add 10 more from Asia')",
    )


def _validate_enrichment_request(data: EnrichTemplateRequest) -> tuple[dict, dict, str]:
    """
    Validate enrichment request and return (template, llm_enrichment, brand_name).

    Raises HTTPException on validation failure.
    """
    templates = _load_policy_templates_from_local_backup()
    template = next((t for t in templates if t.get("id") == data.template_id), None)
    if template is None:
        raise HTTPException(
            status_code=404, detail=f"Template '{data.template_id}' not found"
        )

    llm_enrichment = template.get("llm_enrichment")
    if llm_enrichment is None:
        raise HTTPException(
            status_code=400, detail="Template does not support LLM enrichment"
        )

    # Validate competitors list size if provided
    if data.competitors and len(data.competitors) > MAX_COMPETITOR_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"competitors list exceeds maximum of {MAX_COMPETITOR_NAMES}",
        )

    brand_name = data.parameters.get(llm_enrichment["parameter"], "")
    if not brand_name:
        raise HTTPException(
            status_code=400,
            detail=f"Parameter '{llm_enrichment['parameter']}' is required",
        )

    return template, llm_enrichment, brand_name


@router.post(
    "/policy/templates/enrich",
    tags=["policy management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def enrich_policy_template(
    data: EnrichTemplateRequest,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> dict:
    """
    Enrich a policy template with LLM-discovered data (e.g. competitor names).

    Calls an onboarded LLM to discover competitors for the given brand name,
    then returns enriched guardrailDefinitions with the discovered data populated.
    """
    template, llm_enrichment, brand_name = _validate_enrichment_request(data)
    model = data.model or DEFAULT_COMPETITOR_DISCOVERY_MODEL

    if data.competitors:
        competitors = data.competitors
    else:
        prompt = llm_enrichment["prompt"].replace(
            "{{" + llm_enrichment["parameter"] + "}}", brand_name
        )
        competitors = await _discover_competitors_via_llm(prompt, model=model)

    variations_map = await _generate_competitor_variations(competitors, model=model)

    enriched_definitions = _build_competitor_guardrail_definitions(
        template.get("guardrailDefinitions", []),
        competitors,
        brand_name,
        variations_map,
    )

    return {
        "guardrailDefinitions": enriched_definitions,
        "competitors": competitors,
        "competitor_variations": variations_map,
    }


def _build_refinement_prompt(
    instruction: str,
    existing_competitors: list[str],
    brand_name: str,
) -> str:
    """Build a prompt for refining the competitor list based on user instruction."""
    existing_list = ", ".join(existing_competitors)
    return (
        f"I have a brand called '{brand_name}' and the following competitor list:\n"
        f"{existing_list}\n\n"
        f"User instruction: {instruction}\n\n"
        "Return ONLY the NEW names to add (not the existing ones), one per line, "
        "no numbering, no explanations. If the instruction asks to remove names, "
        "return nothing."
    )


async def _stream_llm_competitor_names(
    prompt: str,
    model: str,
    existing: list[str],
) -> AsyncIterator[tuple[Optional[str], bool]]:
    """
    Stream competitor names from LLM. Yields (name, is_error) tuples.

    Deduplicates against existing names (case-insensitive).
    """
    from litellm.proxy.proxy_server import llm_router

    if llm_router is None:
        raise ValueError("LLM router not initialized")

    existing_lower = {n.lower() for n in existing}
    response = await llm_router.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=COMPETITOR_LLM_TEMPERATURE,
        stream=True,
    )
    buffer = ""
    count = len(existing)
    async for chunk in response:  # type: ignore[union-attr]
        delta = chunk.choices[0].delta.content or ""
        buffer += delta
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            name = _clean_competitor_line(line)
            if (
                name
                and name.lower() not in existing_lower
                and count < MAX_COMPETITOR_NAMES
            ):
                existing_lower.add(name.lower())
                count += 1
                yield name, False
    # Handle remaining buffer
    name = _clean_competitor_line(buffer)
    if name and name.lower() not in existing_lower and count < MAX_COMPETITOR_NAMES:
        yield name, False


async def _stream_competitor_events(
    data: EnrichTemplateRequest,
    template: dict,
    llm_enrichment: dict,
    brand_name: str,
    model: str,
) -> AsyncIterator[str]:
    """Stream competitor names as SSE events, then emit a final 'done' event."""
    competitors: list[str] = list(data.competitors or [])

    if data.instruction and competitors:
        # Refinement mode: keep existing, stream only new names
        for comp in competitors:
            yield f"data: {json.dumps({'type': 'competitor', 'name': comp})}\n\n"

        refinement_prompt = _build_refinement_prompt(
            data.instruction, competitors, brand_name
        )
        try:
            async for name, _ in _stream_llm_competitor_names(
                refinement_prompt, model, competitors
            ):
                if name:
                    competitors.append(name)
                    yield f"data: {json.dumps({'type': 'competitor', 'name': name})}\n\n"
        except Exception as e:
            verbose_proxy_logger.error("LLM competitor refinement failed: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return
    elif data.competitors and not data.instruction:
        # Free-form mode (no instruction): just emit existing
        for comp in competitors:
            yield f"data: {json.dumps({'type': 'competitor', 'name': comp})}\n\n"
    else:
        # Initial discovery mode
        prompt = llm_enrichment["prompt"].replace(
            "{{" + llm_enrichment["parameter"] + "}}", brand_name
        )
        try:
            async for name, _ in _stream_llm_competitor_names(prompt, model, []):
                if name:
                    competitors.append(name)
                    yield f"data: {json.dumps({'type': 'competitor', 'name': name})}\n\n"
        except Exception as e:
            verbose_proxy_logger.error("LLM competitor streaming failed: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

    yield f"data: {json.dumps({'type': 'status', 'message': f'Generating alternate spellings for {len(competitors)} competitors...'})}\n\n"
    variations_map = await _generate_competitor_variations(competitors, model=model)

    total_variations = sum(len(v) for v in variations_map.values())
    yield f"data: {json.dumps({'type': 'status', 'message': f'Building guardrail definitions with {total_variations} variations...'})}\n\n"
    enriched_definitions = _build_competitor_guardrail_definitions(
        template.get("guardrailDefinitions", []),
        competitors,
        brand_name,
        variations_map,
    )

    yield f"data: {json.dumps({'type': 'done', 'competitors': competitors, 'competitor_variations': variations_map, 'guardrailDefinitions': enriched_definitions})}\n\n"


@router.post(
    "/policy/templates/enrich/stream",
    tags=["policy management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def enrich_policy_template_stream(
    data: EnrichTemplateRequest,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Stream competitor names as SSE events as the LLM generates them.

    Events:
    - data: {"type": "competitor", "name": "..."}  — each competitor as discovered
    - data: {"type": "done", "competitors": [...], "competitor_variations": {...}, "guardrailDefinitions": [...]}
    """
    template, llm_enrichment, brand_name = _validate_enrichment_request(data)
    model = data.model or DEFAULT_COMPETITOR_DISCOVERY_MODEL

    return StreamingResponse(
        _stream_competitor_events(data, template, llm_enrichment, brand_name, model),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _clean_competitor_line(line: str) -> Optional[str]:
    """Strip numbering, bullets, and whitespace from a competitor name line."""
    name = line.strip().strip(".-) ").strip()
    return name if name and len(name) > 1 else None


async def _generate_competitor_variations(
    competitors: list, model: str = DEFAULT_COMPETITOR_DISCOVERY_MODEL
) -> dict:
    """Generate common misspellings, abbreviations, and alternate names for each competitor."""
    if not competitors:
        return {}

    # Cap the list to prevent oversized prompts
    capped = competitors[:MAX_COMPETITOR_NAMES]
    names_list = "\n".join(capped)
    prompt = (
        "For each company/brand name below, list 3-5 common misspellings, abbreviations, "
        "and alternate names that people might type. Include typos, missing spaces, "
        "wrong suffixes (e.g. 'Airlines' vs 'Airways' vs 'Airline'), and common shortcuts.\n\n"
        f"Names:\n{names_list}\n\n"
        "Return the result as one line per variation in the format:\n"
        "OriginalName: variation1, variation2, variation3\n"
        "Use the EXACT original name before the colon. No numbering, no extra text."
    )

    try:
        from litellm.proxy.proxy_server import llm_router

        if llm_router is None:
            raise ValueError("LLM router not initialized")
        response = await llm_router.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=COMPETITOR_LLM_TEMPERATURE,
        )
        raw = response.choices[0].message.content or ""  # type: ignore
        return _parse_variations_response(raw, capped)
    except Exception as e:
        verbose_proxy_logger.error("LLM competitor variation generation failed: %s", e)
        return {}


def _parse_variations_response(raw: str, competitors: list) -> dict[str, list[str]]:
    """Parse the LLM response for competitor variations into a name -> variations map."""
    # Build a lowercase lookup for case-insensitive matching
    lower_to_canonical = {comp.lower(): comp for comp in competitors}
    variations_map: dict[str, list[str]] = {}

    for line in raw.strip().split("\n"):
        if ":" not in line:
            continue
        name, _, variations_str = line.partition(":")
        canonical = lower_to_canonical.get(name.strip().lower())
        if canonical is None:
            continue
        variations = [
            v.strip()
            for v in variations_str.split(",")
            if v.strip() and v.strip().lower() != canonical.lower()
        ]
        variations_map[canonical] = variations

    return variations_map


async def _discover_competitors_via_llm(
    prompt: str, model: str = DEFAULT_COMPETITOR_DISCOVERY_MODEL
) -> list:
    """Call an onboarded LLM to discover competitor names."""
    try:
        from litellm.proxy.proxy_server import llm_router

        if llm_router is None:
            raise ValueError("LLM router not initialized")
        response = await llm_router.acompletion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=COMPETITOR_LLM_TEMPERATURE,
        )
        raw = response.choices[0].message.content or ""  # type: ignore
        competitors = [
            name
            for line in raw.strip().split("\n")
            if (name := _clean_competitor_line(line)) is not None
        ]
        return competitors[:MAX_COMPETITOR_NAMES]
    except Exception as e:
        verbose_proxy_logger.error("LLM competitor discovery failed: %s", e)
        return []


def _build_all_names_per_competitor(
    competitors: list[str], variations_map: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Build canonical + variation name lists for each competitor."""
    return {comp: [comp] + variations_map.get(comp, []) for comp in competitors}


def _build_competitor_guardrail_definitions(
    definitions: list,
    competitors: list,
    brand_name: str,
    variations_map: Optional[dict] = None,
) -> list:
    """Build enriched guardrailDefinitions with competitor names and variations populated."""
    variations_map = variations_map or {}
    enriched = copy.deepcopy(definitions)
    all_names = _build_all_names_per_competitor(competitors, variations_map)

    output_blocked = _build_name_blocked_words(competitors, all_names)
    recommendation_blocked = _build_recommendation_blocked_words(competitors, all_names)
    comparison_blocked = _build_comparison_blocked_words(
        competitors, all_names, brand_name
    )

    blocked_words_map = {
        "competitor-output-blocker": output_blocked,
        "competitor-input-blocker": output_blocked,
        "competitor-name-blocker": output_blocked,
        "competitor-name-input-blocker": output_blocked,
        "competitor-name-output-blocker": output_blocked,
        "competitor-recommendation-filter": recommendation_blocked,
        "competitor-recommendation-input-filter": recommendation_blocked,
        "competitor-recommendation-output-filter": recommendation_blocked,
        "competitor-comparison-filter": comparison_blocked,
        "competitor-comparison-input-filter": comparison_blocked,
        "competitor-comparison-output-filter": comparison_blocked,
    }

    for defn in enriched:
        guardrail_name = defn.get("guardrail_name", "")
        if guardrail_name in blocked_words_map:
            defn["litellm_params"]["blocked_words"] = blocked_words_map[guardrail_name]

    return enriched


def _build_name_blocked_words(
    competitors: list[str], all_names: dict[str, list[str]]
) -> list[dict]:
    """Build blocked word entries for direct competitor name mentions."""
    result = []
    for comp in competitors:
        for name in all_names[comp]:
            desc = (
                f"Competitor: {comp}"
                if name == comp
                else f"Competitor variation ({comp}): {name}"
            )
            result.append({"keyword": name, "action": "BLOCK", "description": desc})
    return result


def _build_recommendation_blocked_words(
    competitors: list[str], all_names: dict[str, list[str]]
) -> list[dict]:
    """Build blocked word entries for competitor recommendations."""
    result = []
    for comp in competitors:
        for name in all_names[comp]:
            for prefix in ["try", "use", "switch to", "consider"]:
                result.append(
                    {
                        "keyword": f"{prefix} {name}",
                        "action": "BLOCK",
                        "description": f"Recommendation to competitor ({comp})",
                    }
                )
    return result


def _build_comparison_blocked_words(
    competitors: list[str], all_names: dict[str, list[str]], brand_name: str
) -> list[dict]:
    """Build blocked word entries for unfavorable competitor comparisons."""
    result = []
    for comp in competitors:
        for name in all_names[comp]:
            result.append(
                {
                    "keyword": f"{name} is better",
                    "action": "BLOCK",
                    "description": f"Unfavorable comparison ({comp})",
                }
            )

    # Brand-level comparisons (only need one entry each, not per-competitor)
    result.append(
        {
            "keyword": f"better than {brand_name}",
            "action": "BLOCK",
            "description": "Unfavorable comparison",
        }
    )
    result.append(
        {
            "keyword": f"{brand_name} is worse",
            "action": "BLOCK",
            "description": "Unfavorable comparison",
        }
    )

    return result


class SuggestTemplatesRequest(BaseModel):
    attack_examples: List[str] = Field(default_factory=list)
    description: str = Field(default="")
    model: Optional[str] = None


@router.post(
    "/policy/templates/suggest",
    tags=["policy management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def suggest_policy_templates(
    data: SuggestTemplatesRequest,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> dict:
    """
    Use AI to suggest policy templates based on attack examples and descriptions.

    Calls an LLM with tool calling to match user requirements to available templates.
    """
    from litellm.proxy.management_endpoints.policy_endpoints.ai_policy_suggester import \
        AiPolicySuggester

    templates = _load_policy_templates_from_local_backup()
    suggester = AiPolicySuggester()
    return await suggester.suggest(
        templates=templates,
        attack_examples=data.attack_examples,
        description=data.description,
        model=data.model,
    )


class GuardrailTestResultEntry(TypedDict):
    guardrail_name: str
    action: str  # "passed" | "blocked" | "masked" | "unsupported"
    output_text: str
    details: str


class TestPolicyTemplateRequest(BaseModel):
    guardrail_definitions: List[dict] = Field(
        description="All guardrailDefinitions from the policy template"
    )
    text: str = Field(description="Test input text to run guardrails against")


class TestPolicyTemplateResponse(TypedDict):
    overall_action: str  # worst-case across all guardrails
    results: List[GuardrailTestResultEntry]


@router.post(
    "/policy/templates/test",
    tags=["policy management"],
    dependencies=[Depends(user_api_key_auth)],
)
@management_endpoint_wrapper
async def test_policy_template(
    data: TestPolicyTemplateRequest,
    request: Request,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> TestPolicyTemplateResponse:
    """
    Test a policy template's guardrails against a text input without creating them.

    Instantiates temporary guardrails from the template definitions, runs them
    against the provided text, and returns per-guardrail results so users can
    verify the template solves their problem before creating it.
    """
    from litellm.proxy.utils import handle_exception_on_proxy

    try:
        results = await _test_guardrail_definitions(
            guardrail_definitions=data.guardrail_definitions,
            text=data.text,
        )
        overall = _compute_overall_action(results)
        return TestPolicyTemplateResponse(
            overall_action=overall,
            results=results,
        )
    except Exception as e:
        raise handle_exception_on_proxy(e)


async def _test_guardrail_definitions(
    guardrail_definitions: List[dict],
    text: str,
) -> List[GuardrailTestResultEntry]:
    """Instantiate and run each guardrail definition against the text."""
    from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import \
        ContentFilterGuardrail

    results: List[GuardrailTestResultEntry] = []

    for guardrail_def in guardrail_definitions:
        guardrail_name = guardrail_def.get("guardrail_name", "unknown")
        litellm_params = guardrail_def.get("litellm_params", {})
        guardrail_type = litellm_params.get("guardrail", "")

        if guardrail_type != "litellm_content_filter":
            results.append(
                GuardrailTestResultEntry(
                    guardrail_name=guardrail_name,
                    action="unsupported",
                    output_text=text,
                    details=f"Preview not available for guardrail type: {guardrail_type}",
                )
            )
            continue

        try:
            guardrail = ContentFilterGuardrail(
                guardrail_name=guardrail_name,
                patterns=litellm_params.get("patterns"),
                blocked_words=litellm_params.get("blocked_words"),
                categories=litellm_params.get("categories"),
                pattern_redaction_format=litellm_params.get("pattern_redaction_format"),
                default_on=litellm_params.get("default_on", False),
            )

            output = await guardrail.apply_guardrail(
                inputs={"texts": [text]},
                request_data={},
                input_type="request",
            )
            output_text = (
                output.get("texts", [text])[0] if output.get("texts") else text
            )

            if output_text != text:
                action = "masked"
                details = "Content was modified (masked)"
            else:
                action = "passed"
                details = "No issues detected"

            results.append(
                GuardrailTestResultEntry(
                    guardrail_name=guardrail_name,
                    action=action,
                    output_text=output_text,
                    details=details,
                )
            )
        except HTTPException as e:
            detail = e.detail if hasattr(e, "detail") else str(e)
            if isinstance(detail, dict):
                detail = detail.get("error", str(detail))
            results.append(
                GuardrailTestResultEntry(
                    guardrail_name=guardrail_name,
                    action="blocked",
                    output_text="",
                    details=str(detail),
                )
            )
        except Exception as e:
            results.append(
                GuardrailTestResultEntry(
                    guardrail_name=guardrail_name,
                    action="error",
                    output_text=text,
                    details=str(e),
                )
            )

    return results


def _compute_overall_action(results: List[GuardrailTestResultEntry]) -> str:
    """Return the worst-case action: blocked > masked > error > unsupported > passed."""
    priority = {"blocked": 4, "masked": 3, "error": 2, "unsupported": 1, "passed": 0}
    worst = "passed"
    for r in results:
        if priority.get(r["action"], 0) > priority.get(worst, 0):
            worst = r["action"]
    return worst
