from pytest import Pytester


class TestIsolationFixture:
    CODE = """
        import litellm
        def test_write_cache_a():
            assert litellm.in_memory_llm_clients_cache.get_cache("flushme") is None
            litellm.in_memory_llm_clients_cache.set_cache(key="flushme", value="old_client")
        
        def test_write_cache_b():
            assert litellm.in_memory_llm_clients_cache.get_cache("flushme") is None
            litellm.in_memory_llm_clients_cache.set_cache(key="flushme", value="new_client")
        """

    def test_fixture_isolates(self, testdir: Pytester) -> None:
        testdir.makepyfile(self.CODE)
        result = testdir.runpytest("-p", "litellm.pytest_plugin")
        result.assert_outcomes(passed=2)

    def test_fixture_needed(self, testdir: Pytester) -> None:
        testdir.makepyfile(self.CODE)
        result = testdir.runpytest("-p", "no:litellm.pytest_plugin")
        result.assert_outcomes(passed=1, failed=1)
