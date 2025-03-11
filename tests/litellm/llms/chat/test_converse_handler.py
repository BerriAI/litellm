import os
import sys

from litellm.llms.bedrock.chat import BedrockConverseLLM

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
import litellm

def test_encode_model_id_with_inference_profile():
    """
    Tests to make sure model name is being escaped correctly when used with an inference profile
    :return:
    """
    test = "bedrock/converse/arn:aws:bedrock:us-east-1:12345678910:application-inference-profile/ujdtmcirjhevpi"
    expected = "bedrock/converse/arn%3Aaws%3Abedrock%3Aus-east-1%3A12345678910%3Aapplication-inference-profile%2Fujdtmcirjhevpi"
    got = BedrockConverseLLM.encode_model_id(test)
    assert expected == got
