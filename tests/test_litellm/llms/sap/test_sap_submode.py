from litellm.llms.sap.submode import SapBackendForm, split_sap_submode


def test_plain_model_is_orchestration():
    form, bare = split_sap_submode("anthropic--claude-4.8-opus")
    assert form is SapBackendForm.ORCHESTRATION
    assert bare == "anthropic--claude-4.8-opus"


def test_deployment_prefix_selects_deployment_and_strips_prefix():
    form, bare = split_sap_submode("deployment/anthropic--claude-4.8-opus")
    assert form is SapBackendForm.DEPLOYMENT
    assert bare == "anthropic--claude-4.8-opus"


def test_provider_prefixed_deployment_selects_deployment_and_strips_both_prefixes():
    form, bare = split_sap_submode("sap/deployment/anthropic--claude-4.8-opus")
    assert form is SapBackendForm.DEPLOYMENT
    assert bare == "anthropic--claude-4.8-opus"


def test_provider_prefixed_orchestration_strips_only_provider_prefix():
    form, bare = split_sap_submode("sap/gpt-4o")
    assert form is SapBackendForm.ORCHESTRATION
    assert bare == "gpt-4o"


def test_prefix_must_be_leading_segment():
    form, bare = split_sap_submode("my-deployment/anthropic--claude-4.8-opus")
    assert form is SapBackendForm.ORCHESTRATION
    assert bare == "my-deployment/anthropic--claude-4.8-opus"


def test_bare_prefix_yields_empty_model():
    form, bare = split_sap_submode("deployment/")
    assert form is SapBackendForm.DEPLOYMENT
    assert bare == ""


def test_backend_form_values_are_stable_strings():
    assert SapBackendForm.DEPLOYMENT.value == "deployment"
    assert SapBackendForm.ORCHESTRATION.value == "orchestration"
