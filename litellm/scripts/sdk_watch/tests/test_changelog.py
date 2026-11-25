from datetime import date

from sdk_watch.changelog import ChangelogEntry, parse_changelog

ANTHROPIC: str = """# Changelog

## 0.105.0 (2026-05-28)

Full Changelog: [v0.104.1...v0.105.0](https://github.com/anthropics/anthropic-sdk-python/compare/v0.104.1...v0.105.0)

### Features

* **api:** Add support for claude-opus-4-8, mid-conversation system blocks, and usage.output_tokens_details ([f18b014](https://github.com/anthropics/anthropic-sdk-python/commit/f18b01414b21b49943a6ba2cdaa30ff7dd6a3025))
* support custom file size caps ([#1825](https://github.com/anthropics/anthropic-sdk-python/issues/1825))


### Chores

* **examples:** rename managed-agents private-sandbox-worker to self-hosted-sandbox-worker

## 0.104.1 (2026-05-21)

### Bug Fixes

* **client:** keep the retry
  header on redirects
"""

GENAI: str = """# Changelog

## Unreleased

### Features

* not released yet

## [3.6.1](https://github.com/googleapis/python-genai/compare/v3.6.0...v3.6.1) (2026-11-18)


### Bug Fixes

* Update sdk enum ([4c4ab16](https://github.com/googleapis/python-genai/commit/4c4ab16bc8229415c548519a4cf400cd85936e91))
"""


def test_parses_each_release_with_its_date_and_sectioned_entries() -> None:
    releases = parse_changelog(ANTHROPIC)

    assert [(release.version, release.released) for release in releases] == [
        ("0.105.0", date(2026, 5, 28)),
        ("0.104.1", date(2026, 5, 21)),
    ]
    assert [entry.section for entry in releases[0].entries] == ["Features", "Features", "Chores"]
    assert releases[0].entries[0].text.startswith("**api:** Add support for claude-opus-4-8, mid-conversation system")
    assert "commit/f18b01414b21b49943a6ba2cdaa30ff7dd6a3025" in releases[0].entries[0].text


def test_joins_wrapped_entry_lines_into_one_entry() -> None:
    assert parse_changelog(ANTHROPIC)[1].entries == (
        ChangelogEntry(section="Bug Fixes", text="**client:** keep the retry header on redirects"),
    )


def test_reads_bracketed_release_headings_and_skips_undated_ones() -> None:
    releases = parse_changelog(GENAI)

    assert [(release.version, release.released) for release in releases] == [("3.6.1", date(2026, 11, 18))]
    assert releases[0].entries[0].text.startswith("Update sdk enum")
