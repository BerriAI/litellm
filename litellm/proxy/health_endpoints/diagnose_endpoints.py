import base64
import importlib.metadata
import json
import os
import sys
from typing import Any, Optional

import yaml
from fastapi import APIRouter, Depends, HTTPException, status

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.health_check import run_with_timeout
from litellm.proxy.health_endpoints.diagnose_types import DiagnoseRequest

router = APIRouter()

_DIAGNOSE_REDACTED_VALUE = "[REDACTED]"
_DIAGNOSE_LLM_TIMEOUT_SECONDS = 30
_DIAGNOSE_ADMIN_ROLES = frozenset(
    {
        LitellmUserRoles.PROXY_ADMIN.value,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    }
)
_diagnose_sensitive_masker = SensitiveDataMasker()


def _is_diagnose_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    role = user_api_key_dict.user_role
    if role is None:
        return False
    role_value = role.value if hasattr(role, "value") else role
    return role_value in _DIAGNOSE_ADMIN_ROLES


def _redact_diagnose_payload(payload: Any, sensitive_parent: bool = False) -> Any:
    if isinstance(payload, dict):
        redacted: dict = {}
        for key, value in payload.items():
            key_is_sensitive = _diagnose_sensitive_masker.is_sensitive_key(str(key))
            if key_is_sensitive:
                redacted[key] = _DIAGNOSE_REDACTED_VALUE
            else:
                redacted[key] = _redact_diagnose_payload(value, key_is_sensitive)
        return redacted
    if isinstance(payload, list):
        return [_redact_diagnose_payload(item, sensitive_parent) for item in payload]
    if sensitive_parent:
        return _DIAGNOSE_REDACTED_VALUE
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        return payload
    return str(payload)


def _get_litellm_package_version() -> str:
    try:
        return importlib.metadata.version("litellm")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _get_litellm_installation_info() -> dict:
    try:
        distribution = importlib.metadata.distribution("litellm")
    except importlib.metadata.PackageNotFoundError:
        distribution = None

    installer = distribution.read_text("INSTALLER") if distribution else None
    is_container_runtime = os.path.exists("/.dockerenv") or bool(
        os.getenv("KUBERNETES_SERVICE_HOST")
    )
    package_manager = (installer or "unknown").strip() or "unknown"

    return {
        "runtime": "docker_or_container" if is_container_runtime else "bare_python",
        "is_docker_or_container": is_container_runtime,
        "install_source": (
            "docker_or_container" if is_container_runtime else package_manager
        ),
        "package_manager": package_manager,
        "package_location": str(distribution.locate_file("")) if distribution else None,
        "has_direct_url_metadata": (
            distribution.read_text("direct_url.json") is not None
            if distribution
            else False
        ),
    }


def _dump_diagnose_config_yaml(redacted_config: Any) -> str:
    return yaml.safe_dump(redacted_config, sort_keys=False)


def _build_diagnose_response(
    *,
    used_llm: bool,
    selected_model: Optional[str],
    diagnostic_report: str,
    diagnostic_context: dict,
    next_question: Optional[str] = None,
    next_question_index: Optional[int] = None,
    next_request_body: Optional[dict] = None,
    next_curl: Optional[str] = None,
) -> dict:
    response_context = {
        key: value
        for key, value in diagnostic_context.items()
        if key != "diagnostic_questions"
    }
    return {
        "used_llm": used_llm,
        "selected_model": selected_model,
        "litellm_version": diagnostic_context["litellm_version"],
        "installation": diagnostic_context["installation"],
        "redacted_config_yaml": diagnostic_context["redacted_config_yaml"],
        "next_question": next_question,
        "next_question_index": next_question_index,
        "diagnostic_answers_received": len(
            diagnostic_context.get("diagnostic_answers", [])
        ),
        "next_request_body": next_request_body,
        "next_curl": next_curl,
        "redacted_config": diagnostic_context["config"],
        "diagnostic_report": diagnostic_report,
        "diagnostic_context": response_context,
    }


def _encode_diagnose_questions(questions: list[str]) -> str:
    serialized = json.dumps(questions, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(serialized).decode("ascii")


def _decode_diagnose_questions(session_id: Optional[str]) -> Optional[list[str]]:
    if not session_id:
        return None
    try:
        decoded = base64.urlsafe_b64decode(session_id.encode("ascii"))
        questions = json.loads(decoded.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(questions, list) or not all(
        isinstance(question, str) for question in questions
    ):
        return None
    return questions[:3]


def _build_diagnose_next_request(
    *,
    request_body: dict,
    answers: list[str],
    questions: list[str],
    next_answer_placeholder: str,
) -> dict:
    request_body = {**request_body}
    request_body["diagnostic_session_id"] = _encode_diagnose_questions(questions)
    request_body["diagnostic_answers"] = [*answers, next_answer_placeholder]
    return request_body


def _build_diagnose_next_curl(
    *,
    next_request_body: dict,
) -> str:
    return (
        "curl -X POST http://<your-litellm-proxy>/diagnose \\\n"
        "  -H 'Authorization: Bearer <admin-key>' \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        f"  -d '{json.dumps(next_request_body)}'"
    )


def _build_diagnose_question_report(
    *, next_question_index: int, next_question: str, next_curl: str
) -> str:
    return (
        "# LiteLLM Diagnostic Intake\n\n"
        f"Question {next_question_index} of 3:\n\n"
        f"{next_question}\n\n"
        "Answer it by calling `/diagnose` again with this shape:\n\n"
        f"```bash\n{next_curl}\n```"
    )


def _get_diagnose_model_list(llm_router: Optional[Any]) -> list:
    if llm_router is None:
        return []
    model_list = llm_router.get_model_list(model_name=None)
    return model_list or []


def _select_diagnose_model(
    requested_model: Optional[str], model_list: list
) -> Optional[str]:
    if requested_model:
        return requested_model
    for deployment in model_list:
        if isinstance(deployment, dict) and deployment.get("model_name"):
            return deployment["model_name"]
    return None


def _build_diagnose_prompt(
    *,
    request: DiagnoseRequest,
    selected_model: str,
    diagnostic_context: dict,
) -> str:
    serialized_context = json.dumps(
        diagnostic_context, indent=2, sort_keys=True, default=str
    )
    return (
        "You are helping the LiteLLM team reproduce a proxy issue. "
        "Act like a concise support engineer doing a mini diagnostic grill: "
        "identify the exact LiteLLM version, summarize the configured proxy models, "
        "state whether the deployment appears to be Docker/container or pip-installed, "
        "highlight relevant YAML/config settings, list missing details to ask the admin, "
        "and produce clean Markdown reproduction steps.\n\n"
        f"Model selected for this diagnostic LLM call: {selected_model}\n"
        f"Issue description from admin: {request.issue_description or 'Not provided'}\n"
        f"Known reproduction steps from admin: {request.reproduction_steps or 'Not provided'}\n\n"
        "Diagnostic questions asked:\n"
        f"{json.dumps(diagnostic_context['diagnostic_questions'], indent=2)}\n\n"
        "Admin answers:\n"
        f"{json.dumps(request.diagnostic_answers or [], indent=2)}\n\n"
        "Use only this redacted diagnostic context; never invent secrets:\n"
        f"```json\n{serialized_context}\n```\n\n"
        "Return Markdown with these headings: Summary, Environment, Config and Models, "
        "Reproduction Steps, Questions for Admin, Suspected Areas."
    )


def _build_diagnose_questions_prompt(
    *,
    request: DiagnoseRequest,
    selected_model: str,
    diagnostic_context: dict,
) -> str:
    serialized_context = json.dumps(
        diagnostic_context, indent=2, sort_keys=True, default=str
    )
    return (
        "You are debugging a LiteLLM proxy issue. Generate exactly three concise "
        "questions to ask the admin before writing a reproduction report. The first "
        "question must ask what issue they are seeing. The next two questions should "
        "be follow-ups based on the issue, configured models, LiteLLM version, "
        "installation/runtime, and redacted config. Do not ask for secrets or API keys.\n\n"
        f"Model selected for this diagnostic LLM call: {selected_model}\n"
        f"Known issue description: {request.issue_description or 'Not provided'}\n"
        f"Known reproduction steps: {request.reproduction_steps or 'Not provided'}\n\n"
        "Redacted diagnostic context:\n"
        f"```json\n{serialized_context}\n```\n\n"
        "Return only a numbered list with exactly three questions."
    )


def _parse_diagnose_questions(raw: str) -> list[str]:
    questions: list[str] = []
    for line in raw.splitlines():
        cleaned = line.strip().lstrip("-*").strip()
        if "." in cleaned[:4]:
            _, _, cleaned = cleaned.partition(".")
            cleaned = cleaned.strip()
        if cleaned:
            questions.append(cleaned)
        if len(questions) == 3:
            break
    return questions


def _fallback_diagnose_questions(request: DiagnoseRequest) -> list[str]:
    return [
        "What issue are you seeing in LiteLLM, and what did you expect to happen instead?",
        (
            "Which model, provider, route, and request payload reproduces the issue?"
            if not request.issue_description
            else "Which model, provider, route, and request payload reproduces this issue?"
        ),
        "What errors, logs, status codes, or traces do you see when it fails?",
    ]


async def _generate_diagnose_questions(
    *,
    llm_router: Any,
    selected_model: str,
    request: DiagnoseRequest,
    diagnostic_context: dict,
) -> list[str]:
    response = await run_with_timeout(
        llm_router.acompletion(
            model=selected_model,
            messages=[
                {
                    "role": "user",
                    "content": _build_diagnose_questions_prompt(
                        request=request,
                        selected_model=selected_model,
                        diagnostic_context=diagnostic_context,
                    ),
                }
            ],
            temperature=0.0,
        ),
        _DIAGNOSE_LLM_TIMEOUT_SECONDS,
    )
    questions = _parse_diagnose_questions(_extract_diagnose_response_text(response))
    if len(questions) < 3:
        return _fallback_diagnose_questions(request=request)
    return questions


def _extract_diagnose_response_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    return content or ""


def _build_diagnose_llm_error_report(*, diagnostic_context: dict) -> str:
    return (
        "# LiteLLM Diagnostic Report\n\n"
        "LiteLLM collected the redacted diagnostic context, but the selected "
        "LLM failed while generating the final report. Use the context below "
        "to reproduce the issue manually.\n\n"
        f"```json\n{json.dumps(diagnostic_context, indent=2, sort_keys=True, default=str)}\n```"
    )


@router.post(
    "/diagnose",
    tags=["health"],
    dependencies=[Depends(user_api_key_auth)],
)
async def diagnose_endpoint(
    diagnose_request: DiagnoseRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Guide a proxy admin through collecting a support-ready diagnostic report.
    """
    if not _is_diagnose_admin(user_api_key_dict):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only proxy admins can call /diagnose.",
        )

    from litellm.proxy.proxy_server import llm_router, proxy_config

    configured_models = _get_diagnose_model_list(llm_router=llm_router)
    selected_model = _select_diagnose_model(
        requested_model=diagnose_request.model,
        model_list=configured_models,
    )
    redacted_config = _redact_diagnose_payload(proxy_config.get_config_state())
    redacted_models = _redact_diagnose_payload(configured_models)
    redacted_config_yaml = _dump_diagnose_config_yaml(redacted_config)
    request_dict = diagnose_request.model_dump(exclude_none=True)
    answers = list(diagnose_request.diagnostic_answers or [])
    diagnostic_context = {
        "litellm_version": _get_litellm_package_version(),
        "installation": _get_litellm_installation_info(),
        "python_version": sys.version,
        "configured_models": redacted_models,
        "config": redacted_config,
        "redacted_config_yaml": redacted_config_yaml,
        "diagnostic_questions": [],
        "diagnostic_answers": answers,
        "admin_user": {
            "user_id": user_api_key_dict.user_id,
            "user_role": (
                user_api_key_dict.user_role.value
                if hasattr(user_api_key_dict.user_role, "value")
                else user_api_key_dict.user_role
            ),
        },
    }

    if selected_model is None or llm_router is None:
        diagnostic_context["diagnostic_questions"] = _fallback_diagnose_questions(
            request=diagnose_request
        )
        next_question = diagnostic_context["diagnostic_questions"][0]
        next_answer_placeholder = "<answer question 1 here>"
        next_request_body = _build_diagnose_next_request(
            request_body=request_dict,
            answers=answers,
            questions=diagnostic_context["diagnostic_questions"],
            next_answer_placeholder=next_answer_placeholder,
        )
        next_curl = _build_diagnose_next_curl(next_request_body=next_request_body)
        return _build_diagnose_response(
            used_llm=False,
            selected_model=selected_model,
            diagnostic_report=(
                "# LiteLLM Diagnostic Intake\n\n"
                "No proxy model is configured for the diagnostic LLM call, so "
                "LiteLLM is using built-in fallback questions.\n\n"
                + _build_diagnose_question_report(
                    next_question_index=1,
                    next_question=next_question,
                    next_curl=next_curl,
                )
            ),
            diagnostic_context=diagnostic_context,
            next_question=next_question,
            next_question_index=1,
            next_request_body=next_request_body,
            next_curl=next_curl,
        )

    diagnostic_questions = _decode_diagnose_questions(
        diagnose_request.diagnostic_session_id
    )
    if diagnostic_questions is None:
        try:
            diagnostic_questions = await _generate_diagnose_questions(
                llm_router=llm_router,
                selected_model=selected_model,
                request=diagnose_request,
                diagnostic_context=diagnostic_context,
            )
        except Exception as e:
            verbose_proxy_logger.warning(
                "Failed to generate /diagnose questions with LLM: %s", e
            )
            diagnostic_questions = _fallback_diagnose_questions(
                request=diagnose_request
            )
    diagnostic_context["diagnostic_questions"] = diagnostic_questions

    diagnostic_answers_count = len(answers)
    if diagnostic_answers_count < 3:
        next_question_index = diagnostic_answers_count + 1
        next_question = diagnostic_questions[diagnostic_answers_count]
        next_answer_placeholder = f"<answer question {next_question_index} here>"
        next_request_body = _build_diagnose_next_request(
            request_body=request_dict,
            answers=answers,
            questions=diagnostic_questions,
            next_answer_placeholder=next_answer_placeholder,
        )
        next_curl = _build_diagnose_next_curl(next_request_body=next_request_body)
        return _build_diagnose_response(
            used_llm=True,
            selected_model=selected_model,
            diagnostic_report=_build_diagnose_question_report(
                next_question_index=next_question_index,
                next_question=next_question,
                next_curl=next_curl,
            ),
            diagnostic_context=diagnostic_context,
            next_question=next_question,
            next_question_index=next_question_index,
            next_request_body=next_request_body,
            next_curl=next_curl,
        )

    prompt = _build_diagnose_prompt(
        request=diagnose_request,
        selected_model=selected_model,
        diagnostic_context=diagnostic_context,
    )
    try:
        response = await run_with_timeout(
            llm_router.acompletion(
                model=selected_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            ),
            _DIAGNOSE_LLM_TIMEOUT_SECONDS,
        )
    except Exception as e:
        verbose_proxy_logger.warning(
            "Failed to generate /diagnose report with LLM: %s", e
        )
        return _build_diagnose_response(
            used_llm=False,
            selected_model=selected_model,
            diagnostic_report=_build_diagnose_llm_error_report(
                diagnostic_context=diagnostic_context,
            ),
            diagnostic_context=diagnostic_context,
        )
    return _build_diagnose_response(
        used_llm=True,
        selected_model=selected_model,
        diagnostic_report=_extract_diagnose_response_text(response),
        diagnostic_context=diagnostic_context,
    )
