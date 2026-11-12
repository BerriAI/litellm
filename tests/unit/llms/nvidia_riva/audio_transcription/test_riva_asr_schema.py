from collections.abc import Iterator
from typing import Final

from google.protobuf.descriptor import FieldDescriptor
from riva.client.proto import riva_asr_pb2, riva_audio_pb2

from litellm.llms.nvidia_riva.audio_transcription import riva_asr_schema


def _declared_fields() -> Iterator[tuple[str, FieldDescriptor]]:
    for message_name, _ in riva_asr_schema._MESSAGES:
        message: Final = riva_asr_schema._POOL.FindMessageTypeByName(f"{riva_asr_schema._PACKAGE}.{message_name}")
        yield from ((message_name, field) for field in message.fields)


def _wire_shape(field: FieldDescriptor | None) -> tuple[object, ...] | None:
    if field is None:
        return None
    return (
        field.number,
        field.type,
        field.label,
        field.has_presence,
        field.message_type.full_name if field.message_type else None,
        field.enum_type.name if field.enum_type else None,
        field.containing_oneof.name if field.containing_oneof else None,
    )


def _nvidia_field(message_name: str, field_name: str) -> FieldDescriptor | None:
    return riva_asr_pb2.DESCRIPTOR.message_types_by_name[message_name].fields_by_name.get(field_name)


def test_every_declared_field_matches_nvidias_riva_schema() -> None:
    declared: Final = tuple(_declared_fields())

    assert len(declared) == sum(len(fields) for _, fields in riva_asr_schema._MESSAGES)
    assert {f"{message}.{field.name}": _wire_shape(field) for message, field in declared} == {
        f"{message}.{field.name}": _wire_shape(_nvidia_field(message, field.name)) for message, field in declared
    }
    assert riva_asr_schema.LINEAR_PCM == riva_audio_pb2.AudioEncoding.LINEAR_PCM
