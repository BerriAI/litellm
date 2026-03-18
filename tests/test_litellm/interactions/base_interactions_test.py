"""
Abstract base class for Interactions API tests.

This class provides common test cases that can be inherited by provider-specific
test classes. Subclasses must implement get_model() and get_api_key().
"""

import os
from abc import ABC, abstractmethod

import pytest

import litellm.interactions as interactions


class BaseInteractionsTest(ABC):
    """Abstract base class for interactions API tests.
    
    Subclasses must implement get_model() and get_api_key().
    All test methods are inherited and run against the specific provider.
    """
    
    @abstractmethod
    def get_model(self) -> str:
        """Return the model string for this provider."""
        pass
    
    @abstractmethod
    def get_api_key(self) -> str:
        """Return the API key for this provider."""
        pass
    
    def test_create_simple_string_input(self):
        """Test creating an interaction with a simple string input."""
        api_key = self.get_api_key()
        if not api_key:
            pytest.skip(f"API key not set for {self.__class__.__name__}")
        
        response = interactions.create(
            model=self.get_model(),
            input="Hello, what is 2 + 2?",
            api_key=api_key,
        )
        assert response is not None
        assert response.id is not None or response.status is not None
        
        # Check outputs per OpenAPI spec
        if response.outputs:
            assert len(response.outputs) > 0
        
        # Check usage per OpenAPI spec
        if response.usage:
            # Usage is a dict in InteractionsAPIResponse
            if isinstance(response.usage, dict):
                # Check for both possible key formats: input_tokens/output_tokens or total_input_tokens/total_output_tokens
                assert (
                    response.usage.get("input_tokens") is not None 
                    or response.usage.get("output_tokens") is not None
                    or response.usage.get("total_input_tokens") is not None
                    or response.usage.get("total_output_tokens") is not None
                )
            else:
                # If it's an object, check attributes
                assert hasattr(response.usage, "input_tokens") or hasattr(response.usage, "output_tokens")
    
    def test_create_with_system_instruction(self):
        """Test creating an interaction with system_instruction."""
        api_key = self.get_api_key()
        if not api_key:
            pytest.skip(f"API key not set for {self.__class__.__name__}")
        
        response = interactions.create(
            model=self.get_model(),
            input="What are you?",
            system_instruction="You are a helpful pirate assistant. Always respond like a pirate.",
            api_key=api_key,
        )
        assert response is not None
        # Verify the response reflects the system instruction
        if response.outputs:
            assert len(response.outputs) > 0
    
    def test_create_streaming(self):
        """Test creating a streaming interaction."""
        api_key = self.get_api_key()
        if not api_key:
            pytest.skip(f"API key not set for {self.__class__.__name__}")
        
        response_stream = interactions.create(
            model=self.get_model(),
            input="Count from 1 to 3.",
            stream=True,
            api_key=api_key,
        )
        
        # Collect all chunks
        chunks = []
        for chunk in response_stream:
            chunks.append(chunk)
        
        assert len(chunks) > 0
    
    @pytest.mark.asyncio
    async def test_acreate_simple(self):
        """Test async interaction creation."""
        api_key = self.get_api_key()
        if not api_key:
            pytest.skip(f"API key not set for {self.__class__.__name__}")
        
        response = await interactions.acreate(
            model=self.get_model(),
            input="What is the speed of light?",
            api_key=api_key,
        )
        assert response is not None
        assert response.id is not None or response.status is not None

