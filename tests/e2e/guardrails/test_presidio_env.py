"""Harness coverage for the Presidio endpoint scrubber.

`scrub` is what keeps an internal endpoint out of a CI log when a presidio test
fails, and nothing downstream would notice if it quietly stopped replacing
anything, so its two pure halves are covered directly. No proxy, no e2e marker.
"""

from __future__ import annotations

from presidio_env import REDACTED, redact, secret_fragments

ANALYZER = "https://analyzer.internal.example/"
ANONYMIZER = "https://anonymizer.internal.example/"
BASES = (ANALYZER, ANONYMIZER)


class TestSecretFragments:
    def test_each_endpoint_contributes_its_url_slashless_form_and_bare_host(self) -> None:
        fragments = secret_fragments(BASES)

        assert set(fragments) == {
            ANALYZER,
            ANALYZER.rstrip("/"),
            "analyzer.internal.example",
            ANONYMIZER,
            ANONYMIZER.rstrip("/"),
            "anonymizer.internal.example",
        }

    def test_fragments_are_longest_first(self) -> None:
        """A host is a substring of its own URL, so replacing the short form first
        would leave the scheme and path of the long form behind in the log."""
        lengths = [len(fragment) for fragment in secret_fragments(BASES)]

        assert lengths == sorted(lengths, reverse=True)


class TestRedact:
    def test_a_full_url_is_replaced_whole(self) -> None:
        redacted = redact(f"Making request to: {ANALYZER}analyze", secret_fragments(BASES))

        assert redacted == f"Making request to: {REDACTED}analyze"
        assert "internal.example" not in redacted

    def test_a_bare_host_is_replaced_too(self) -> None:
        """aiohttp names only the host when it cannot connect, so a connection
        error is the shape most likely to carry the endpoint into a log."""
        redacted = redact("Cannot connect to host anonymizer.internal.example:443", secret_fragments(BASES))

        assert redacted == f"Cannot connect to host {REDACTED}:443"

    def test_every_occurrence_goes_not_just_the_first(self) -> None:
        redacted = redact(f"{ANALYZER} then {ANALYZER}", secret_fragments(BASES))

        assert redacted == f"{REDACTED} then {REDACTED}"

    def test_text_carrying_no_endpoint_is_returned_unchanged(self) -> None:
        assert redact("Presidio PII analysis failed: ClientConnectorError", secret_fragments(BASES)) == (
            "Presidio PII analysis failed: ClientConnectorError"
        )
