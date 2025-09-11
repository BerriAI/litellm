from .._encoding import BufferedEncoder
from ..encoding import MSGPACK_ENCODERS


class WriterClientBase(object):
    """A class encapsulating an endpoint/encoder pair that a TraceWriter can send payloads to"""

    ENDPOINT = ""

    def __init__(
        self,
        encoder: BufferedEncoder,
    ):
        self.encoder = encoder


class AgentWriterClientV5(WriterClientBase):
    ENDPOINT = "v0.5/traces"

    def __init__(self, buffer_size, max_payload_size):
        super(AgentWriterClientV5, self).__init__(
            MSGPACK_ENCODERS["v0.5"](
                max_size=buffer_size,
                max_item_size=max_payload_size,
            )
        )


class AgentWriterClientV4(WriterClientBase):
    ENDPOINT = "v0.4/traces"

    def __init__(self, buffer_size, max_payload_size):
        super(AgentWriterClientV4, self).__init__(
            MSGPACK_ENCODERS["v0.4"](
                max_size=buffer_size,
                max_item_size=max_payload_size,
            )
        )


WRITER_CLIENTS = {
    "v0.4": AgentWriterClientV4,
    "v0.5": AgentWriterClientV5,
}
