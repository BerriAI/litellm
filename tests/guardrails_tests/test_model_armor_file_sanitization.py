import sys
import os
import pytest
from unittest.mock import AsyncMock
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy.guardrails.guardrail_hooks.model_armor.model_armor import ModelArmorGuardrail

def test_sanitize_file_prompt_builds_pdf_body():
	guardrail = ModelArmorGuardrail(
		template_id="dummy-template",
		project_id="dummy-project",
		location="us-central1",
		credentials=None,
	)
	file_bytes = b"%PDF-1.4 some pdf content"
	file_type = "PDF"
	body = guardrail.sanitize_file_prompt(file_bytes, file_type, source="user_prompt")
	assert "userPromptData" in body
	assert body["userPromptData"]["byteItem"]["byteDataType"] == "PDF"
	import base64
	assert body["userPromptData"]["byteItem"]["byteData"] == base64.b64encode(file_bytes).decode("utf-8")

@pytest.mark.asyncio
async def test_make_model_armor_request_file_prompt():
	guardrail = ModelArmorGuardrail(
		template_id="dummy-template",
		project_id="dummy-project",
		location="us-central1",
		credentials=None,
	)
	file_bytes = b"My SSN is 123-45-6789."
	file_type = "PLAINTEXT_UTF8"
	armor_response = {
		"sanitizationResult": {
			"filterResults": [
				{
					"sdpFilterResult": {
						"inspectResult": {
							"executionState": "EXECUTION_SUCCESS",
							"matchState": "MATCH_FOUND",
							"findings": [
								{"infoType": "US_SOCIAL_SECURITY_NUMBER", "likelihood": "LIKELY"}
							]
						},
						"deidentifyResult": {
							"executionState": "EXECUTION_SUCCESS",
							"matchState": "MATCH_FOUND",
							"data": {"text": "My SSN is [REDACTED]."}
						}
					}
				}
			]
		}
	}
	class MockResponse:
		def __init__(self, status_code, text, json_data):
			self.status_code = status_code
			self.text = text
			self._json = json_data
		def json(self):
			return self._json
	class MockHandler:
		async def post(self, url, json, headers):
			return MockResponse(200, str(armor_response), armor_response)
	guardrail.async_handler = MockHandler()
	guardrail._ensure_access_token_async = AsyncMock(return_value=("dummy-token", "dummy-project"))
	result = await guardrail.make_model_armor_request(
		file_bytes=file_bytes,
		file_type=file_type,
		source="user_prompt"
	)
	assert result["sanitizationResult"]["filterResults"][0]["sdpFilterResult"]["deidentifyResult"]["data"]["text"] == "My SSN is [REDACTED]."
