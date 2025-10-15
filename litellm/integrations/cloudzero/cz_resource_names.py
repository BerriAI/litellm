# Copyright 2025 CloudZero
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# CHANGELOG: 2025-01-19 - Initial CZRN module for CloudZero Resource Names (erik.peterson)

"""CloudZero Resource Names (CZRN) generation and validation for LiteLLM resources."""

import re
from enum import Enum
from typing import Any, cast

import litellm


class CZEntityType(str, Enum):
    TEAM = "team"


class CZRNGenerator:
    """Generate CloudZero Resource Names (CZRNs) for LiteLLM resources."""

    CZRN_REGEX = re.compile(r'^czrn:([a-z0-9-]+):([a-zA-Z0-9-]+):([a-z0-9-]+):([a-z0-9-]+):([a-z0-9-]+):(.+)$')

    def __init__(self):
        """Initialize CZRN generator."""
        pass

    def create_from_litellm_data(self, row: dict[str, Any]) -> str:
        """Create a CZRN from LiteLLM daily spend data.
        
        CZRN format: czrn:<service-type>:<provider>:<region>:<owner-account-id>:<resource-type>:<cloud-local-id>
        
        For LiteLLM resources, we map:
        - service-type: 'litellm' (the service managing the LLM calls)
        - provider: The custom_llm_provider (e.g., 'openai', 'anthropic', 'azure')
        - region: 'cross-region' (LiteLLM operates across regions)
        - owner-account-id: The team_id or user_id (entity_id)
        - resource-type: 'llm-usage' (represents LLM usage/inference)
        - cloud-local-id: model
        """
        service_type = 'litellm'
        provider = self._normalize_provider(row.get('custom_llm_provider', 'unknown'))
        region = 'cross-region'

        # Use the actual entity_id (team_id or user_id) as the owner account
        team_id = row.get('team_id', 'unknown')
        owner_account_id = self._normalize_component(team_id)

        resource_type = 'llm-usage'

        # Create a unique identifier with just the model (entity info already in owner_account_id)
        model = row.get('model', 'unknown')

        cloud_local_id = model

        return self.create_from_components(
            service_type=service_type,
            provider=provider,
            region=region,
            owner_account_id=owner_account_id,
            resource_type=resource_type,
            cloud_local_id=cloud_local_id
        )

    def create_from_components(
        self,
        service_type: str,
        provider: str,
        region: str,
        owner_account_id: str,
        resource_type: str,
        cloud_local_id: str
    ) -> str:
        """Create a CZRN from individual components."""
        # Normalize components to ensure they meet CZRN requirements
        service_type = self._normalize_component(service_type, allow_uppercase=True)
        provider = self._normalize_component(provider)
        region = self._normalize_component(region)
        owner_account_id = self._normalize_component(owner_account_id)
        resource_type = self._normalize_component(resource_type)
        # cloud_local_id can contain pipes and other characters, so don't normalize it

        czrn = f"czrn:{service_type}:{provider}:{region}:{owner_account_id}:{resource_type}:{cloud_local_id}"

        if not self.is_valid(czrn):
            raise ValueError(f"Generated CZRN is invalid: {czrn}")

        return czrn

    def is_valid(self, czrn: str) -> bool:
        """Validate a CZRN string against the standard format."""
        return bool(self.CZRN_REGEX.match(czrn))

    def extract_components(self, czrn: str) -> tuple[str, str, str, str, str, str]:
        """Extract all components from a CZRN.
        
        Returns: (service_type, provider, region, owner_account_id, resource_type, cloud_local_id)
        """
        match = self.CZRN_REGEX.match(czrn)
        if not match:
            raise ValueError(f"Invalid CZRN format: {czrn}")

        return cast(tuple[str, str, str, str, str, str], match.groups())

    def _normalize_provider(self, provider: str) -> str:
        """Normalize provider names to standard CZRN format."""
        # Map common provider names to CZRN standards
        provider_map = {
            litellm.LlmProviders.AZURE.value: 'azure',
            litellm.LlmProviders.AZURE_AI.value: 'azure',
            litellm.LlmProviders.ANTHROPIC.value: 'anthropic',
            litellm.LlmProviders.BEDROCK.value: 'aws',
            litellm.LlmProviders.VERTEX_AI.value: 'gcp',
            litellm.LlmProviders.GEMINI.value: 'google',
            litellm.LlmProviders.COHERE.value: 'cohere',
            litellm.LlmProviders.HUGGINGFACE.value: 'huggingface',
            litellm.LlmProviders.REPLICATE.value: 'replicate',
            litellm.LlmProviders.TOGETHER_AI.value: 'together-ai',
        }

        normalized = provider.lower().replace('_', '-')

        # use litellm custom llm provider if not in provider_map
        if normalized not in provider_map:
            return normalized
        return provider_map.get(normalized, normalized)

    def _normalize_component(self, component: str, allow_uppercase: bool = False) -> str:
        """Normalize a CZRN component to meet format requirements."""
        if not component:
            return 'unknown'

        # Convert to lowercase unless uppercase is allowed
        if not allow_uppercase:
            component = component.lower()

        # Replace invalid characters with hyphens
        component = re.sub(r'[^a-zA-Z0-9-]', '-', component)

        # Remove consecutive hyphens
        component = re.sub(r'-+', '-', component)

        # Remove leading/trailing hyphens
        component = component.strip('-')

        return component or 'unknown'

