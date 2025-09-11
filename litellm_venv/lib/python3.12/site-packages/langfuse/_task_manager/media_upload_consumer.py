import logging
import threading

from .media_manager import MediaManager


class MediaUploadConsumer(threading.Thread):
    _log = logging.getLogger(__name__)
    _identifier: int
    _max_retries: int
    _media_manager: MediaManager

    def __init__(
        self,
        *,
        identifier: int,
        media_manager: MediaManager,
    ):
        """Create a consumer thread."""
        super().__init__()
        # Make consumer a daemon thread so that it doesn't block program exit
        self.daemon = True
        # It's important to set running in the constructor: if we are asked to
        # pause immediately after construction, we might set running to True in
        # run() *after* we set it to False in pause... and keep running
        # forever.
        self.running = True
        self._identifier = identifier
        self._media_manager = media_manager

    def run(self):
        """Run the media upload consumer."""
        self._log.debug("consumer is running...")
        while self.running:
            self._media_manager.process_next_media_upload()

    def pause(self):
        """Pause the media upload consumer."""
        self.running = False
