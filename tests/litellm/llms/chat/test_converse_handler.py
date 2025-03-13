import os
import sys

from litellm.llms.bedrock.chat import BedrockConverseLLM

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
import litellm


def test_encode_model_id_with_inference_profile():
    """
    Test instance profile is properly encoded when used as a model
    """
    test_model = "arn:aws:bedrock:us-east-1:12345678910:application-inference-profile/ujdtmcirjhevpi"
    expected_model = "arn%3Aaws%3Abedrock%3Aus-east-1%3A12345678910%3Aapplication-inference-profile%2Fujdtmcirjhevpi"
    bedrock_converse_llm = BedrockConverseLLM()
    returned_model = bedrock_converse_llm.encode_model_id(test_model)
    assert expected_model == returned_model
