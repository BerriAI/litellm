import argparse
from typing import Final


def main() -> int:
    parser: Final = argparse.ArgumentParser(description="Report the ledger cutover rollback boundary")
    parser.add_argument("--all-writers-ledger-disabled", action="store_true")
    parser.add_argument("--ledger-buffers-drained", action="store_true")
    parser.parse_args()
    print("Blocked: ledger cutover ended the rollback window; the legacy writer can no longer be enabled")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
