import sys
from pathlib import Path

if __name__ == "__main__":
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tests.sdk_function_trace.runtime import main

    main("python")
