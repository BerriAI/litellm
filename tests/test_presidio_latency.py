
import asyncio
import aiohttp
import pytest
from unittest.mock import MagicMock, patch
from litellm.proxy.guardrails.guardrail_hooks.presidio import _OPTIONAL_PresidioPIIMasking

@pytest.mark.asyncio
async def test_sanity_presidio_session_reuse_main_thread():
    """
    SANITY CHECK:
    Verify that Presidio guardrail reuses sessions in the main thread.
    This ensures we don't break existing session pooling functionality.
    """
    presidio = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_analyzer_api_base="http://mock-analyzer",
        presidio_anonymizer_api_base="http://mock-anonymizer"
    )
    
    session_creations = 0
    original_init = aiohttp.ClientSession.__init__
    
    def mocked_init(self, *args, **kwargs):
        nonlocal session_creations
        session_creations += 1
        original_init(self, *args, **kwargs)

    with patch.object(aiohttp.ClientSession, "__init__", side_effect=mocked_init, autospec=True):
        for _ in range(10):
            async with presidio._get_session_iterator() as session:
                pass
        
        # Expected: Only 1 session created for all 10 calls.
        assert session_creations == 1
        
    await presidio._close_http_session()

@pytest.mark.asyncio
async def test_bug_presidio_session_explosion_background_thread_causes_latency():
    """
    BUG REPRODUCTION:
    Verify that background threads (like logging hooks) REUSE sessions.
    Previously, each call in a background loop created a NEW ephemeral session, 
    leading to socket exhaustion and the reported 97s latency spike.
    """
    import threading
    presidio = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_analyzer_api_base="http://mock-analyzer",
        presidio_anonymizer_api_base="http://mock-anonymizer"
    )
    
    # Force the code to think it's in a background thread
    presidio._main_thread_id = threading.get_ident() + 1
    
    session_creations = 0
    original_init = aiohttp.ClientSession.__init__
    
    def mocked_init(self, *args, **kwargs):
        nonlocal session_creations
        session_creations += 1
        original_init(self, *args, **kwargs)

    with patch.object(aiohttp.ClientSession, "__init__", side_effect=mocked_init, autospec=True):
        for _ in range(10):
            async with presidio._get_session_iterator() as session:
                pass
        
        # FIX VERIFICATION: Should now be 1 session (reused) instead of 10.
        assert session_creations == 1
        
    await presidio._close_http_session()

@pytest.mark.asyncio
async def test_optimization_presidio_avoid_recognizer_reloads_on_server():
    """
    OPTIMIZATION VERIFICATION:
    Verify that ad_hoc_recognizers are OMITTED from the payload when 
    presidio_ad_hoc_recognizers_on_server is True.
    Sending them on every request forces the Presidio server to reload its 
    entire registry, spiking CPU and latency.
    """
    presidio = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_analyzer_api_base="http://mock-analyzer",
        presidio_anonymizer_api_base="http://mock-anonymizer",
        presidio_ad_hoc_recognizers_on_server=True
    )
    presidio.ad_hoc_recognizers = [{"name": "CustomRecognizer", "supported_entity": "CUSTOM"}]

    payload = presidio._get_presidio_analyze_request_payload(
        text="some text",
        presidio_config=None,
        request_data={}
    )

    # Optimization Check: payload should NOT contain the redundant recognizers
    assert "ad_hoc_recognizers" not in payload

@pytest.mark.asyncio
async def test_optimization_presidio_pii_caching_skips_network_calls():
    """
    OPTIMIZATION VERIFICATION:
    Verify that PII results are cached in local memory.
    Identical text should return instantly from cache without making a 
    network round-trip to the Presidio service.
    """
    presidio = _OPTIONAL_PresidioPIIMasking(
        mock_testing=True,
        presidio_analyzer_api_base="http://mock-analyzer",
        presidio_anonymizer_api_base="http://mock-anonymizer"
    )
    
    # Mock network calls
    presidio.analyze_text = MagicMock(return_value=asyncio.Future())
    presidio.analyze_text.return_value.set_result([])
    presidio.anonymize_text = MagicMock(return_value=asyncio.Future())
    presidio.anonymize_text.return_value.set_result("redacted text")
    
    # First call - Must hit the mock (Network)
    result1 = await presidio.check_pii(
        text="repetitive call center text",
        output_parse_pii=False,
        presidio_config=None,
        request_data={}
    )
    assert result1 == "redacted text"
    assert presidio.analyze_text.call_count == 1
    
    # Second call - Must hit the CACHE (No Network)
    result2 = await presidio.check_pii(
        text="repetitive call center text",
        output_parse_pii=False,
        presidio_config=None,
        request_data={}
    )
    assert result2 == "redacted text"
    assert presidio.analyze_text.call_count == 1 # Counter should NOT increase
