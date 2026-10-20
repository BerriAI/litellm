from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

MUSE_SPARK_MODEL = "meta/muse-spark-1.1"


def test_muse_spark_1_1_model_info():
    routed_model, provider, _, api_base = get_llm_provider(model=MUSE_SPARK_MODEL, api_key="sk-test")
    assert routed_model == "muse-spark-1.1"
    assert provider == "meta"
    assert api_base == "https://api.meta.ai/v1"
