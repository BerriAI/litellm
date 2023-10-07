from typing import Optional, Union
import types

# This file just has the openai config classes. 
# For implementation check out completion() in main.py

class OpenAIConfig():
    """
    Reference: https://platform.openai.com/docs/api-reference/chat/create

    The class `OpenAIConfig` provides configuration for the OpenAI's Chat API interface. Below are the parameters:

    - `frequency_penalty` (number or null): Defaults to 0. Allows a value between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, thereby minimizing repetition.

    - `function_call` (string or object): This optional parameter controls how the model calls functions.

    - `functions` (array): An optional parameter. It is a list of functions for which the model may generate JSON inputs.

    - `logit_bias` (map): This optional parameter modifies the likelihood of specified tokens appearing in the completion.

    - `max_tokens` (integer or null): This optional parameter helps to set the maximum number of tokens to generate in the chat completion.

    - `n` (integer or null): This optional parameter helps to set how many chat completion choices to generate for each input message.

    - `presence_penalty` (number or null): Defaults to 0. It penalizes new tokens based on if they appear in the text so far, hence increasing the model's likelihood to talk about new topics.

    - `stop` (string / array / null): Specifies up to 4 sequences where the API will stop generating further tokens.

    - `temperature` (number or null): Defines the sampling temperature to use, varying between 0 and 2.

    - `top_p` (number or null): An alternative to sampling with temperature, used for nucleus sampling. 
    """
    frequency_penalty: Optional[int]=None
    function_call: Optional[Union[str, dict]]=None
    functions: Optional[list]=None
    logit_bias: Optional[dict]=None
    max_tokens: Optional[int]=None
    n: Optional[int]=None
    presence_penalty: Optional[int]=None
    stop: Optional[Union[str, list]]=None
    temperature: Optional[int]=None
    top_p: Optional[int]=None

    def __init__(self,
                 frequency_penalty: Optional[int]=None,
                 function_call: Optional[Union[str, dict]]=None,
                 functions: Optional[list]=None,
                 logit_bias: Optional[dict]=None,
                 max_tokens: Optional[int]=None,
                 n: Optional[int]=None,
                 presence_penalty: Optional[int]=None,
                 stop: Optional[Union[str, list]]=None,
                 temperature: Optional[int]=None,
                 top_p: Optional[int]=None,) -> None:
        
        locals_ = locals()
        for key, value in locals_.items():
            if key != 'self' and value is not None:
                setattr(self.__class__, key, value)
   
    @classmethod
    def get_config(cls):
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') 
                and not isinstance(v, (types.FunctionType, types.BuiltinFunctionType, classmethod, staticmethod)) 
                and v is not None}

class OpenAITextCompletionConfig():
    """
    Reference: https://platform.openai.com/docs/api-reference/completions/create

    The class `OpenAITextCompletionConfig` provides configuration for the OpenAI's text completion API interface. Below are the parameters:

    - `best_of` (integer or null): This optional parameter generates server-side completions and returns the one with the highest log probability per token.

    - `echo` (boolean or null): This optional parameter will echo back the prompt in addition to the completion.

    - `frequency_penalty` (number or null): Defaults to 0. It is a numbers from -2.0 to 2.0, where positive values decrease the model's likelihood to repeat the same line.

    - `logit_bias` (map): This optional parameter modifies the likelihood of specified tokens appearing in the completion.
    
    - `logprobs` (integer or null): This optional parameter includes the log probabilities on the most likely tokens as well as the chosen tokens.

    - `max_tokens` (integer or null): This optional parameter sets the maximum number of tokens to generate in the completion.

    - `n` (integer or null): This optional parameter sets how many completions to generate for each prompt.

    - `presence_penalty` (number or null): Defaults to 0 and can be between -2.0 and 2.0. Positive values increase the model's likelihood to talk about new topics.
    
    - `stop` (string / array / null): Specifies up to 4 sequences where the API will stop generating further tokens.
    
    - `suffix` (string or null): Defines the suffix that comes after a completion of inserted text.

    - `temperature` (number or null): This optional parameter defines the sampling temperature to use.
    
    - `top_p` (number or null): An alternative to sampling with temperature, used for nucleus sampling.
    """
    best_of: Optional[int]=None
    echo: Optional[bool]=None
    frequency_penalty: Optional[int]=None
    logit_bias: Optional[dict]=None
    logprobs: Optional[int]=None
    max_tokens: Optional[int]=None
    n: Optional[int]=None
    presence_penalty: Optional[int]=None
    stop: Optional[Union[str, list]]=None
    suffix: Optional[str]=None
    temperature: Optional[float]=None
    top_p: Optional[float]=None

    def __init__(self,
                 best_of: Optional[int]=None,
                 echo: Optional[bool]=None,
                 frequency_penalty: Optional[int]=None,
                 logit_bias: Optional[dict]=None,
                 logprobs: Optional[int]=None,
                 max_tokens: Optional[int]=None,
                 n: Optional[int]=None,
                 presence_penalty: Optional[int]=None,
                 stop: Optional[Union[str, list]]=None,
                 suffix: Optional[str]=None,
                 temperature: Optional[float]=None,
                 top_p: Optional[float]=None) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != 'self' and value is not None:
                setattr(self.__class__, key, value)
   
    @classmethod
    def get_config(cls):
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('__') 
                and not isinstance(v, (types.FunctionType, types.BuiltinFunctionType, classmethod, staticmethod)) 
                and v is not None}


class AzureOpenAIConfig(OpenAIConfig):
    """
    Reference: https://platform.openai.com/docs/api-reference/chat/create

    The class `AzureOpenAIConfig` provides configuration for the OpenAI's Chat API interface, for use with Azure. It inherits from `OpenAIConfig`. Below are the parameters::

    - `frequency_penalty` (number or null): Defaults to 0. Allows a value between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, thereby minimizing repetition.

    - `function_call` (string or object): This optional parameter controls how the model calls functions.

    - `functions` (array): An optional parameter. It is a list of functions for which the model may generate JSON inputs.

    - `logit_bias` (map): This optional parameter modifies the likelihood of specified tokens appearing in the completion.

    - `max_tokens` (integer or null): This optional parameter helps to set the maximum number of tokens to generate in the chat completion.

    - `n` (integer or null): This optional parameter helps to set how many chat completion choices to generate for each input message.

    - `presence_penalty` (number or null): Defaults to 0. It penalizes new tokens based on if they appear in the text so far, hence increasing the model's likelihood to talk about new topics.

    - `stop` (string / array / null): Specifies up to 4 sequences where the API will stop generating further tokens.

    - `temperature` (number or null): Defines the sampling temperature to use, varying between 0 and 2.

    - `top_p` (number or null): An alternative to sampling with temperature, used for nucleus sampling. 
    """

    def __init__(self, 
                 frequency_penalty: Optional[int] = None, 
                 function_call: Optional[Union[str, dict]]= None, 
                 functions: Optional[list]= None, 
                 logit_bias: Optional[dict]= None, 
                 max_tokens: Optional[int]= None, 
                 n: Optional[int]= None, 
                 presence_penalty: Optional[int]= None, 
                 stop: Optional[Union[str,list]]=None, 
                 temperature: Optional[int]= None, 
                 top_p: Optional[int]= None) -> None:
        super().__init__(frequency_penalty, 
                         function_call, 
                         functions, 
                         logit_bias, 
                         max_tokens, 
                         n, 
                         presence_penalty, 
                         stop, 
                         temperature, 
                         top_p)