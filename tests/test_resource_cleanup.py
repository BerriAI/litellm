"""
Test that async HTTP clients are properly cleaned up to prevent resource leaks.
Issue: https://github.com/BerriAI/litellm/issues/12107
"""
import asyncio
import os
import warnings

import pytest

import litellm


@pytest.mark.asyncio
async def test_acompletion_resource_cleanup():
    """Test that acompletion doesn't leave unclosed client sessions."""
    # Suppress warnings to check for them later
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Make an async completion call
        response = await litellm.acompletion(
            model="gemini/gemini-2.0-flash-lite-001",
            messages=[{"role": "user", "content": "Hello"}],
            mock_response="Hi there! How can I help you today?",
        )

        # Check that response was received
        assert (
            response.choices[0].message.content == "Hi there! How can I help you today?"
        )

        # Manually close async clients
        await litellm.close_litellm_async_clients()

        # Give a small delay for any warnings to appear
        await asyncio.sleep(0.1)

        # Check for resource warnings
        resource_warnings = [
            warning
            for warning in w
            if "Unclosed" in str(warning.message)
            and (
                "client session" in str(warning.message)
                or "connector" in str(warning.message)
            )
        ]

        # Should be no unclosed resource warnings
        assert (
            len(resource_warnings) == 0
        ), f"Found unclosed resources: {[str(w.message) for w in resource_warnings]}"


@pytest.mark.asyncio
async def test_multiple_acompletion_calls_cleanup():
    """Test that multiple acompletion calls reuse clients and don't leak resources."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Make multiple async completion calls
        for i in range(3):
            response = await litellm.acompletion(
                model="gemini/gemini-2.0-flash-lite-001",
                messages=[{"role": "user", "content": f"Hello {i}"}],
                mock_response=f"Response {i}",
            )
            assert response.choices[0].message.content == f"Response {i}"

        # Clean up
        await litellm.close_litellm_async_clients()

        # Give a small delay for any warnings to appear
        await asyncio.sleep(0.1)

        # Check for resource warnings
        resource_warnings = [
            warning
            for warning in w
            if "Unclosed" in str(warning.message)
            and (
                "client session" in str(warning.message)
                or "connector" in str(warning.message)
            )
        ]

        assert (
            len(resource_warnings) == 0
        ), f"Found unclosed resources: {[str(w.message) for w in resource_warnings]}"


@pytest.mark.asyncio
async def test_cleanup_function_is_safe_to_call_multiple_times():
    """Test that the cleanup function can be called multiple times safely."""
    # This should not raise any errors
    await litellm.close_litellm_async_clients()
    await litellm.close_litellm_async_clients()
    await litellm.close_litellm_async_clients()

    # Should still work after multiple cleanups
    response = await litellm.acompletion(
        model="gemini/gemini-2.0-flash-lite-001",
        messages=[{"role": "user", "content": "Hello"}],
        mock_response="Hi!",
    )
    assert response.choices[0].message.content == "Hi!"

    # Clean up again
    await litellm.close_litellm_async_clients()


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_acompletion_resource_cleanup())
    print("✅ All tests passed!")
