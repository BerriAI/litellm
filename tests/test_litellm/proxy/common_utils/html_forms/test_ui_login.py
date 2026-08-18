import os
import sys

sys.path.insert(0, os.path.abspath("../../../"))

from litellm.proxy.common_utils.html_forms.ui_login import build_ui_login_form

DISCLOSURE_MARKERS = ("Default Credentials", "MASTER_KEY")
FORM_MARKERS = ('name="username"', 'name="password"')


def test_build_ui_login_form_shows_disclosure_by_default():
    html = build_ui_login_form()

    for marker in DISCLOSURE_MARKERS:
        assert marker in html
    for marker in FORM_MARKERS:
        assert marker in html


def test_build_ui_login_form_hides_disclosure_when_flag_set():
    html = build_ui_login_form(hide_default_credentials_hint=True)

    for marker in DISCLOSURE_MARKERS:
        assert marker not in html
    # the login form itself must remain functional, only the hint is removed
    for marker in FORM_MARKERS:
        assert marker in html


def test_build_ui_login_form_hint_independent_of_deprecation_banner():
    with_banner = build_ui_login_form(
        show_deprecation_banner=True, hide_default_credentials_hint=True
    )
    without_banner = build_ui_login_form(
        show_deprecation_banner=False, hide_default_credentials_hint=True
    )

    assert "Deprecated:" in with_banner
    assert "Deprecated:" not in without_banner
    for html in (with_banner, without_banner):
        assert "Default Credentials" not in html
