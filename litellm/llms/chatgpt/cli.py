"""CLI entrypoint for logging in to ChatGPT / Codex backend via OAuth."""

import argparse
import sys

from .authenticator import Authenticator
from .common_utils import ChatGPTAuthError
from .pkce import REDIRECT_PORT as PKCE_DEFAULT_PORT


def cli() -> int:
    parser = argparse.ArgumentParser(
        prog="litellm-chatgpt-login",
        description=(
            "Sign in to the ChatGPT / Codex backend and store OAuth credentials "
            "for use with `litellm.responses(model='chatgpt/...')`."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["device", "pkce"],
        default="device",
        help=(
            "OAuth flow to use. `device` (default) shows a code to enter in a "
            "browser (works over SSH). `pkce` opens a browser to a loopback "
            "redirect (one-click, requires a local browser)."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PKCE_DEFAULT_PORT,
        help="Loopback port for the PKCE redirect (pkce only).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not try to auto-open a browser for the PKCE flow.",
    )
    args = parser.parse_args()

    auth = Authenticator()
    try:
        if args.method == "pkce":
            auth.login_pkce(open_browser=not args.no_browser, port=args.port)
        else:
            auth._login_device_code()
    except ChatGPTAuthError as exc:
        print(f"Login failed: {exc.message}", file=sys.stderr)  # noqa: T201
        return 1
    except KeyboardInterrupt:
        print("\nLogin cancelled.", file=sys.stderr)  # noqa: T201
        return 130
    except Exception as exc:  # noqa: BLE001
        # Any other failure mode (network, filesystem, etc.) — surface a
        # clean message instead of a traceback.
        print(f"Login failed: {exc}", file=sys.stderr)  # noqa: T201
        return 1

    print(f"Saved credentials to {auth.auth_file}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(cli())
