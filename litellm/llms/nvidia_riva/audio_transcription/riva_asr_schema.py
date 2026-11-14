"""
The part of NVIDIA Riva's speech recognition gRPC schema that LiteLLM sends and reads.

Names and numbers match ``riva/proto/riva_asr.proto`` and ``riva/proto/riva_audio.proto`` in
https://github.com/nvidia-riva/common. Fields LiteLLM never sets or reads are left out, and the
wire format skips them. The messages live in their own descriptor pool so they never collide
with nvidia-riva-client's copies when both are imported in one process.
"""

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Final

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from google.protobuf.message import Message

_PACKAGE: Final = "nvidia.riva.asr"
_SERVICE: Final = f"/{_PACKAGE}.RivaSpeechRecognition"
RECOGNIZE_METHOD: Final = f"{_SERVICE}/Recognize"
STREAMING_RECOGNIZE_METHOD: Final = f"{_SERVICE}/StreamingRecognize"


class _Kind(Enum):
    FLOAT = "TYPE_FLOAT"
    INT32 = "TYPE_INT32"
    BOOL = "TYPE_BOOL"
    STRING = "TYPE_STRING"
    MESSAGE = "TYPE_MESSAGE"
    BYTES = "TYPE_BYTES"
    ENUM = "TYPE_ENUM"


class AudioEncoding(IntEnum):
    ENCODING_UNSPECIFIED = 0
    LINEAR_PCM = 1


LINEAR_PCM: Final = AudioEncoding.LINEAR_PCM.value


@dataclass(frozen=True, slots=True)
class _Field:
    name: str
    number: int
    kind: _Kind
    type_name: str = ""
    repeated: bool = False
    optional: bool = False
    oneof: str = ""


_MESSAGES: Final[tuple[tuple[str, tuple[_Field, ...]], ...]] = (
    (
        "RecognitionConfig",
        (
            _Field("encoding", 1, _Kind.ENUM, "AudioEncoding"),
            _Field("sample_rate_hertz", 2, _Kind.INT32),
            _Field("language_code", 3, _Kind.STRING),
            _Field("max_alternatives", 4, _Kind.INT32),
            _Field("profanity_filter", 5, _Kind.BOOL),
            _Field("audio_channel_count", 7, _Kind.INT32),
            _Field("enable_word_time_offsets", 8, _Kind.BOOL),
            _Field("enable_automatic_punctuation", 11, _Kind.BOOL),
            _Field("model", 13, _Kind.STRING),
            _Field("verbatim_transcripts", 14, _Kind.BOOL),
            _Field("endpointing_config", 25, _Kind.MESSAGE, "EndpointingConfig", optional=True),
        ),
    ),
    (
        "EndpointingConfig",
        (
            _Field("start_history", 1, _Kind.INT32, optional=True),
            _Field("start_threshold", 2, _Kind.FLOAT, optional=True),
            _Field("stop_history", 3, _Kind.INT32, optional=True),
            _Field("stop_threshold", 4, _Kind.FLOAT, optional=True),
            _Field("stop_history_eou", 5, _Kind.INT32, optional=True),
            _Field("stop_threshold_eou", 6, _Kind.FLOAT, optional=True),
        ),
    ),
    (
        "StreamingRecognitionConfig",
        (
            _Field("config", 1, _Kind.MESSAGE, "RecognitionConfig"),
            _Field("interim_results", 2, _Kind.BOOL),
        ),
    ),
    (
        "StreamingRecognizeRequest",
        (
            _Field("streaming_config", 1, _Kind.MESSAGE, "StreamingRecognitionConfig", oneof="streaming_request"),
            _Field("audio_content", 2, _Kind.BYTES, oneof="streaming_request"),
        ),
    ),
    (
        "WordInfo",
        (
            _Field("start_time", 1, _Kind.INT32),
            _Field("end_time", 2, _Kind.INT32),
            _Field("word", 3, _Kind.STRING),
        ),
    ),
    (
        "SpeechRecognitionAlternative",
        (
            _Field("transcript", 1, _Kind.STRING),
            _Field("words", 3, _Kind.MESSAGE, "WordInfo", repeated=True),
        ),
    ),
    (
        "StreamingRecognitionResult",
        (
            _Field("alternatives", 1, _Kind.MESSAGE, "SpeechRecognitionAlternative", repeated=True),
            _Field("is_final", 2, _Kind.BOOL),
        ),
    ),
    (
        "StreamingRecognizeResponse",
        (_Field("results", 1, _Kind.MESSAGE, "StreamingRecognitionResult", repeated=True),),
    ),
    (
        "RecognizeRequest",
        (
            _Field("config", 1, _Kind.MESSAGE, "RecognitionConfig"),
            _Field("audio", 2, _Kind.BYTES),
        ),
    ),
    (
        "SpeechRecognitionResult",
        (_Field("alternatives", 1, _Kind.MESSAGE, "SpeechRecognitionAlternative", repeated=True),),
    ),
    (
        "RecognizeResponse",
        (_Field("results", 1, _Kind.MESSAGE, "SpeechRecognitionResult", repeated=True),),
    ),
)


def _oneof_names(fields: tuple[_Field, ...]) -> tuple[str, ...]:
    declared: Final = tuple(dict.fromkeys(field.oneof for field in fields if field.oneof))
    synthetic: Final = tuple(f"_{field.name}" for field in fields if field.optional)
    return declared + synthetic


def _field_proto(field: _Field, oneofs: tuple[str, ...]) -> descriptor_pb2.FieldDescriptorProto:
    oneof: Final = f"_{field.name}" if field.optional else field.oneof
    return descriptor_pb2.FieldDescriptorProto(
        name=field.name,
        number=field.number,
        type=descriptor_pb2.FieldDescriptorProto.Type.Value(field.kind.value),
        label=(
            descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
            if field.repeated
            else descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
        ),
        type_name=f".{_PACKAGE}.{field.type_name}" if field.type_name else None,
        oneof_index=oneofs.index(oneof) if oneof else None,
        proto3_optional=field.optional or None,
    )


def _message_proto(name: str, fields: tuple[_Field, ...]) -> descriptor_pb2.DescriptorProto:
    oneofs: Final = _oneof_names(fields)
    return descriptor_pb2.DescriptorProto(
        name=name,
        field=[_field_proto(field, oneofs) for field in fields],
        oneof_decl=[descriptor_pb2.OneofDescriptorProto(name=oneof) for oneof in oneofs],
    )


_FILE: Final = descriptor_pb2.FileDescriptorProto(
    name="litellm/nvidia_riva/riva_asr.proto",
    package=_PACKAGE,
    syntax="proto3",
    enum_type=[
        descriptor_pb2.EnumDescriptorProto(
            name="AudioEncoding",
            value=[
                descriptor_pb2.EnumValueDescriptorProto(name=encoding.name, number=encoding.value)
                for encoding in AudioEncoding
            ],
        )
    ],
    message_type=[_message_proto(name, fields) for name, fields in _MESSAGES],
)
_POOL: Final = descriptor_pool.DescriptorPool()
_POOL.AddSerializedFile(_FILE.SerializeToString())


def _message_class(name: str) -> type[Message]:
    return message_factory.GetMessageClass(_POOL.FindMessageTypeByName(f"{_PACKAGE}.{name}"))


RecognitionConfig: Final = _message_class("RecognitionConfig")
EndpointingConfig: Final = _message_class("EndpointingConfig")
StreamingRecognitionConfig: Final = _message_class("StreamingRecognitionConfig")
StreamingRecognizeRequest: Final = _message_class("StreamingRecognizeRequest")
StreamingRecognizeResponse: Final = _message_class("StreamingRecognizeResponse")
RecognizeRequest: Final = _message_class("RecognizeRequest")
RecognizeResponse: Final = _message_class("RecognizeResponse")
