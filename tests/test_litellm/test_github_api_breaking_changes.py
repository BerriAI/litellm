"""Unit tests for the public-API gate (`.github/scripts/check_api_breaking_changes.py`).

The griffe-loading half needs a git repo, so these cover the decision half: what
counts as a declaration, what is in scope, and which combinations of findings and
PR metadata are allowed through.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / ".github" / "scripts" / "check_api_breaking_changes.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_api_breaking_changes", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_module()


def finding(kind: str = "OBJECT_REMOVED", path: str = "litellm.BedrockLLM"):
    return gate.ApiFinding(kind=kind, path=path, detail=f"{path}: {kind}", file="litellm/x.py", line=1)


def delta(blocking=(), added=(), advisory=()):
    return gate.ApiDelta(blocking=tuple(blocking), advisory=tuple(advisory), added_names=tuple(added))


class FakeObject:
    """Stands in for a griffe Object/Alias: `path` is where it is exported from,
    `canonical` is where it actually lives (None mimics an unresolvable alias)."""

    def __init__(self, path: str, canonical: str | None, target: str | None = None, lineno: int = 7):
        self.path = path
        self.filepath = Path("litellm/x.py")
        self.lineno = lineno
        self._canonical = canonical
        self._target = target

    @property
    def canonical_path(self) -> str:
        if self._canonical is None:
            raise RuntimeError("alias does not resolve")
        return self._canonical

    @property
    def target_path(self) -> str:
        if self._target is None:
            raise AttributeError("target_path")
        return self._target


class TestParseDeclaration:
    def test_bang_after_type_declares_breaking(self):
        assert gate.parse_declaration("feat!: drop BedrockLLM", "").breaking is True

    def test_bang_after_scope_declares_breaking(self):
        declaration = gate.parse_declaration("refactor(proxy)!: rename hook", "")
        assert declaration.breaking is True
        assert declaration.commit_type == "refactor"

    def test_plain_type_does_not_declare_breaking(self):
        declaration = gate.parse_declaration("feat: add provider", "")
        assert declaration.breaking is False
        assert declaration.commit_type == "feat"

    def test_footer_declares_breaking_without_bang(self):
        body = "Some context\n\nBREAKING CHANGE: `BedrockLLM` is gone, use `BedrockConverse`\n"
        assert gate.parse_declaration("fix: tidy", body).breaking is True

    def test_hyphenated_footer_declares_breaking(self):
        assert gate.parse_declaration("fix: tidy", "BREAKING-CHANGE: gone\n").breaking is True

    def test_footer_must_start_a_line(self):
        body = "We considered whether this is a BREAKING CHANGE: it is not.\n"
        assert gate.parse_declaration("fix: tidy", body).breaking is False

    def test_bang_in_subject_does_not_count(self):
        assert gate.parse_declaration("fix: this is urgent!: really", "").breaking is False

    def test_unparseable_title_has_no_type(self):
        declaration = gate.parse_declaration("Drop BedrockLLM", "")
        assert declaration.commit_type is None
        assert declaration.breaking is False


class TestDecide:
    def test_clean_delta_is_approved(self):
        verdict = gate.decide(delta(), gate.parse_declaration("chore: tidy", ""))
        assert isinstance(verdict, gate.Approved)

    def test_undeclared_breaking_change_is_rejected(self):
        verdict = gate.decide(delta(blocking=[finding()]), gate.parse_declaration("fix: tidy", ""))
        assert isinstance(verdict, gate.UndeclaredBreakingChanges)
        assert verdict.findings[0].path == "litellm.BedrockLLM"

    def test_declared_breaking_change_is_approved(self):
        verdict = gate.decide(delta(blocking=[finding()]), gate.parse_declaration("feat!: drop it", ""))
        assert isinstance(verdict, gate.Approved)

    def test_footer_alone_clears_a_breaking_change(self):
        declaration = gate.parse_declaration("fix: tidy", "BREAKING CHANGE: gone\n")
        assert isinstance(gate.decide(delta(blocking=[finding()]), declaration), gate.Approved)

    def test_advisory_findings_never_block(self):
        only_advisory = delta(advisory=[finding(kind="ATTRIBUTE_CHANGED_VALUE")])
        verdict = gate.decide(only_advisory, gate.parse_declaration("chore: tidy", ""))
        assert isinstance(verdict, gate.Approved)

    @pytest.mark.parametrize("commit_type", ["feat", "fix"])
    def test_new_names_allowed_under_feature_types(self, commit_type: str):
        verdict = gate.decide(delta(added=["new_flag"]), gate.parse_declaration(f"{commit_type}: x", ""))
        assert isinstance(verdict, gate.Approved)

    @pytest.mark.parametrize("commit_type", ["chore", "refactor", "docs", "test", "ci"])
    def test_new_names_rejected_under_non_feature_types(self, commit_type: str):
        verdict = gate.decide(delta(added=["new_flag"]), gate.parse_declaration(f"{commit_type}: x", ""))
        assert isinstance(verdict, gate.UndeclaredSurfaceWidening)
        assert verdict.names == ("new_flag",)
        assert verdict.commit_type == commit_type

    def test_new_names_rejected_when_title_is_unparseable(self):
        verdict = gate.decide(delta(added=["new_flag"]), gate.parse_declaration("Add a flag", ""))
        assert isinstance(verdict, gate.UndeclaredSurfaceWidening)
        assert verdict.commit_type is None

    def test_breaking_change_is_reported_before_surface_widening(self):
        both = delta(blocking=[finding()], added=["new_flag"])
        verdict = gate.decide(both, gate.parse_declaration("chore: tidy", ""))
        assert isinstance(verdict, gate.UndeclaredBreakingChanges)

    def test_breaking_bang_does_not_excuse_surface_widening_under_chore(self):
        both = delta(blocking=[finding()], added=["new_flag"])
        verdict = gate.decide(both, gate.parse_declaration("chore!: tidy", ""))
        assert isinstance(verdict, gate.UndeclaredSurfaceWidening)


class TestScope:
    def test_sdk_object_is_in_scope(self):
        obj = FakeObject("litellm.BedrockLLM", "litellm.llms.bedrock.chat.handler.BedrockLLM")
        assert gate.is_in_scope(obj, "litellm") is True

    def test_proxy_internals_are_out_of_scope(self):
        obj = FakeObject(
            "litellm.proxy.hooks.parallel_request_limiter_v3.TPM_RESERVED_TOKENS_KEY",
            "litellm.proxy.hooks.parallel_request_limiter_v3.TPM_RESERVED_TOKENS_KEY",
        )
        assert gate.is_in_scope(obj, "litellm") is False

    def test_reexported_stdlib_name_is_out_of_scope(self):
        assert gate.is_in_scope(FakeObject("litellm.utils.Union", "typing.Union"), "litellm") is False

    def test_unresolvable_stdlib_alias_is_out_of_scope(self):
        obj = FakeObject("litellm.utils.List", canonical=None, target="typing.List")
        assert gate.is_in_scope(obj, "litellm") is False

    def test_unresolvable_internal_alias_stays_in_scope(self):
        obj = FakeObject("litellm.BedrockLLM", canonical=None, target="litellm.llms.bedrock.BedrockLLM")
        assert gate.is_in_scope(obj, "litellm") is True

    def test_alias_chained_through_an_internal_module_to_stdlib_is_out_of_scope(self):
        chained = FakeModule("Final", external=("Final",))
        assert gate.is_in_scope(chained.members["Final"], "litellm", (chained,)) is False

    def test_alias_chained_within_the_package_stays_in_scope(self):
        owned = FakeModule("completion")
        assert gate.is_in_scope(owned.members["completion"], "litellm", (owned,)) is True

    def test_unidentifiable_alias_is_treated_as_out_of_scope(self):
        assert gate.is_in_scope(FakeObject("litellm.Router", canonical=None), "litellm") is False


class FakeKind:
    def __init__(self, name: str):
        self.name = name


class FakeBreakage:
    def __init__(self, obj: FakeObject, kind: str, explanation: str):
        self.obj = obj
        self.kind = FakeKind(kind)
        self._explanation = explanation

    def explain(self, style: object) -> str:
        return self._explanation


class FakeModule:
    """Top-level `litellm`. Names are owned by the package unless `external` says otherwise.

    `external` names mimic the real `litellm.Final`: re-exported through an
    internal module, so only following the alias chain reveals they are stdlib.
    """

    name = "litellm"

    def __init__(self, *names: str, external: tuple[str, ...] = ()):
        self.members = {
            name: FakeObject(f"litellm.{name}", canonical=None, target=f"litellm.scheduler.{name}")
            if name in external
            else FakeObject(f"litellm.{name}", f"litellm.main.{name}")
            for name in names
        }
        self._hops = {
            f"scheduler.{name}": FakeObject(f"litellm.scheduler.{name}", canonical=None, target=f"typing.{name}")
            for name in external
        }

    def get_member(self, parts):
        return self._hops[".".join(parts)]


class TestBuildDelta:
    def test_griffe_color_codes_are_stripped(self):
        breakage = FakeBreakage(
            FakeObject("litellm.BedrockLLM", "litellm.llms.bedrock.BedrockLLM"),
            "OBJECT_REMOVED",
            "\x1b[1mlitellm/x.py\x1b[0m:0: BedrockLLM: \x1b[33mPublic object was removed\x1b[39m",
        )
        built = gate.build_delta(FakeModule(), FakeModule(), iter([breakage]), style=None)
        assert built.blocking[0].detail == "litellm/x.py:0: BedrockLLM: Public object was removed"
        assert built.blocking[0].line == 7

    def test_the_same_breakage_seen_through_two_aliases_is_reported_once(self):
        duplicate = [
            FakeBreakage(FakeObject("litellm.BedrockLLM", "litellm.llms.bedrock.BedrockLLM"), "OBJECT_REMOVED", "gone"),
            FakeBreakage(FakeObject("litellm.BedrockLLM", "litellm.llms.bedrock.BedrockLLM"), "OBJECT_REMOVED", "gone"),
        ]
        built = gate.build_delta(FakeModule(), FakeModule(), iter(duplicate), style=None)
        assert len(built.blocking) == 1

    def test_out_of_scope_breakages_are_dropped(self):
        noise = [
            FakeBreakage(FakeObject("litellm.utils.Union", "typing.Union"), "OBJECT_REMOVED", "typing gone"),
            FakeBreakage(FakeObject("litellm.proxy.utils.KEY", "litellm.proxy.utils.KEY"), "OBJECT_REMOVED", "k gone"),
        ]
        built = gate.build_delta(FakeModule(), FakeModule(), iter(noise), style=None)
        assert built.blocking == ()

    def test_value_changes_land_in_advisory_not_blocking(self):
        breakage = FakeBreakage(
            FakeObject("litellm.router.Span", "litellm.router.Span"), "ATTRIBUTE_CHANGED_VALUE", "Union[A, B] -> A | B"
        )
        built = gate.build_delta(FakeModule(), FakeModule(), iter([breakage]), style=None)
        assert built.blocking == ()
        assert len(built.advisory) == 1

    def test_added_top_level_names_are_detected_and_private_ones_ignored(self):
        built = gate.build_delta(
            FakeModule("completion"), FakeModule("completion", "new_flag", "_private"), iter([]), style=None
        )
        assert built.added_names == ("new_flag",)

    def test_removed_top_level_names_are_not_counted_as_additions(self):
        built = gate.build_delta(FakeModule("completion", "old_flag"), FakeModule("completion"), iter([]), style=None)
        assert built.added_names == ()

    def test_a_newly_imported_stdlib_name_is_not_surface_widening(self):
        built = gate.build_delta(
            FakeModule("completion"),
            FakeModule("completion", "Final", external=("Final",)),
            iter([]),
            style=None,
        )
        assert built.added_names == ()

    def test_owned_additions_survive_alongside_ignored_stdlib_imports(self):
        built = gate.build_delta(
            FakeModule("completion"),
            FakeModule("completion", "Final", "new_flag", external=("Final",)),
            iter([]),
            style=None,
        )
        assert built.added_names == ("new_flag",)


class TestRendering:
    def test_summary_names_the_declaration_escape_hatch(self):
        blocking = delta(blocking=[finding()])
        summary = gate.render_summary(blocking, gate.decide(blocking, gate.parse_declaration("fix: x", "")))
        assert "BREAKING CHANGE:" in summary
        assert "litellm.BedrockLLM" in summary

    def test_summary_separates_advisory_from_blocking(self):
        mixed = delta(blocking=[finding()], advisory=[finding(kind="ATTRIBUTE_CHANGED_VALUE")])
        summary = gate.render_summary(mixed, gate.decide(mixed, gate.parse_declaration("feat!: x", "")))
        assert "### Breaking changes" in summary
        assert "### Advisory" in summary

    def test_clean_summary_has_no_finding_sections(self):
        summary = gate.render_summary(delta(), gate.Approved())
        assert "###" not in summary

    def test_annotations_point_at_the_source_line(self):
        rendered = gate.render_annotations([finding()])
        assert rendered == "::error file=litellm/x.py,line=1::litellm.BedrockLLM: OBJECT_REMOVED"

    def test_annotation_without_a_location_still_renders(self):
        located = gate.ApiFinding(kind="OBJECT_REMOVED", path="litellm.X", detail="gone", file=None, line=None)
        assert gate.render_annotations([located]) == "::error::gone"
