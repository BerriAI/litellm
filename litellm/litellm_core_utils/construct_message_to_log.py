from io import StringIO
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import litellm
from litellm.litellm_core_utils.audio_utils.utils import FileTypes, get_audio_file_name
from litellm.types.utils import CallTypes

if TYPE_CHECKING:
    from litellm.litellm_core_utils.rules import Rules


def construct_message_to_log(
    call_type: Union[str, CallTypes], args: Union[Tuple[Any, ...], List[Any]], kwargs: Dict[str, Any], rules_obj: "Rules"
) -> Optional[Any]:
    """
    Constructs the messages to log for the given call type.


    Args:
        call_type: The type of call to log
        args: The arguments to log
        kwargs: The keyword arguments to log
        rules_obj: The rules object instance to use for pre-call rules

    Returns:
        The messages to log (can be list, string, or None)
    """
    messages: Optional[Any] = None
    if (
        call_type == CallTypes.completion.value
        or call_type == CallTypes.acompletion.value
        or call_type == CallTypes.anthropic_messages.value
    ):
        messages = None
        if len(args) > 1:
            messages = args[1]
        elif kwargs.get("messages", None):
            messages = kwargs["messages"]
        ### PRE-CALL RULES ###
        from litellm.litellm_core_utils.rules import Rules
        
        if (
            Rules.has_pre_call_rules()
            and isinstance(messages, list)
            and len(messages) > 0
            and isinstance(messages[0], dict)
            and "content" in messages[0]
        ):

            buffer = StringIO()
            for m in messages:
                content = m.get("content", "")
                if content is not None and isinstance(content, str):
                    buffer.write(content)

            model = args[0] if len(args) > 0 else kwargs.get("model", "")
            if model:
                rules_obj.pre_call_rules(
                    input=buffer.getvalue(),
                    model=model,
                )
    elif (
        call_type == CallTypes.embedding.value
        or call_type == CallTypes.aembedding.value
    ):
        messages = args[1] if len(args) > 1 else kwargs.get("input", None)
    elif (
        call_type == CallTypes.image_generation.value
        or call_type == CallTypes.aimage_generation.value
    ):
        messages = args[0] if len(args) > 0 else kwargs["prompt"]
    elif (
        call_type == CallTypes.moderation.value
        or call_type == CallTypes.amoderation.value
    ):
        messages = args[1] if len(args) > 1 else kwargs["input"]
    elif (
        call_type == CallTypes.atext_completion.value
        or call_type == CallTypes.text_completion.value
    ):
        messages = args[0] if len(args) > 0 else kwargs["prompt"]
    elif (
        call_type == CallTypes.rerank.value or call_type == CallTypes.arerank.value
    ):
        messages = kwargs.get("query")
    elif (
        call_type == CallTypes.atranscription.value
        or call_type == CallTypes.transcription.value
    ):
        _file_obj: FileTypes = args[1] if len(args) > 1 else kwargs["file"]
        file_checksum = get_audio_file_name(file_obj=_file_obj)
        if "metadata" in kwargs:
            kwargs["metadata"]["file_checksum"] = file_checksum
        else:
            kwargs["metadata"] = {"file_checksum": file_checksum}
        messages = file_checksum
    elif (
        call_type == CallTypes.aspeech.value or call_type == CallTypes.speech.value
    ):
        messages = kwargs.get("input", "speech")
    elif (
        call_type == CallTypes.aresponses.value
        or call_type == CallTypes.responses.value
    ):
        messages = args[0] if len(args) > 0 else kwargs["input"]
    elif (
        call_type == CallTypes.asearch.value
        or call_type == CallTypes.search.value
    ):
        messages = kwargs.get("query")
    else:
        messages = "default-message-value"
    
    return messages