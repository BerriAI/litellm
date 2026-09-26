from litellm.llms.turing_engine import TuringConfig


def test_turing_config():
    config = TuringConfig(
        temperature=1,
        sparsity_ratio=0.57,
        max_tokens=1024,
        stream=True,
        use_svd_kv=True,
    )
    assert config.temperature == 1
    assert config.sparsity_ratio == 0.57
    assert config.max_tokens == 1024
    assert config.stream is True
    assert config.use_svd_kv is True
    base, key = config._get_openai_compatible_provider_info(None, None)
    assert base == "http://localhost:8000/v1"
    assert key == "turing-local"
    base_custom, key_custom = config._get_openai_compatible_provider_info(
        "http://gpu-node:8000/v1", "secret-key"
    )
    assert base_custom == "http://gpu-node:8000/v1"
    assert key_custom == "secret-key"
