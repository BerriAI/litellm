import signal
import sys
from types import FrameType


def _exit_on_reraised_term(signum: int, frame: FrameType | None) -> None:
    sys.exit(0)


def main() -> None:
    from litellm import run_server

    signal.signal(signal.SIGTERM, _exit_on_reraised_term)
    run_server()


if __name__ == "__main__":
    main()
