"""
Sarvam Audio Response wrapper

Since Sarvam returns base64-encoded audio in JSON (not binary stream),
we need a custom response wrapper that mimics HttpxBinaryResponseContent.
"""


class SarvamAudioResponse:
    """
    Wrapper for Sarvam TTS audio response.

    Mimics the interface of HttpxBinaryResponseContent for compatibility.
    """

    def __init__(self, content: bytes):
        self._content = content
        self._hidden_params: dict = {}

    @property
    def content(self) -> bytes:
        """Return the audio content as bytes."""
        return self._content

    def read(self) -> bytes:
        """Read the audio content."""
        return self._content

    def iter_bytes(self, chunk_size: int = 1024):
        """Iterate over the audio content in chunks."""
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i:i + chunk_size]

    def __iter__(self):
        """Iterate over the audio content."""
        return self.iter_bytes()
