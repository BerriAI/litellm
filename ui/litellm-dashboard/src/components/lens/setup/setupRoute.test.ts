import { createSerializer } from "nuqs";
import { describe, expect, it } from "vitest";
import { investigationDefaults } from "./investigationSchema";
import { draftFromParams, paramsFromDraft, SETUP_DRAFT_PARSERS, type SetupDraftParams } from "./setupRoute";

const blank: SetupDraftParams = {
  q: "",
  name: null,
  lookback: null,
  sample: null,
  max: null,
  context: null,
  watch: null,
  checks: null,
  monitor: null,
  every: null,
};

describe("setup draft params", () => {
  it("opens a blank draft from an empty URL and keeps an untouched draft out of it", () => {
    const draft = draftFromParams(blank);
    expect(draft).toEqual(investigationDefaults(undefined, "new"));
    expect(paramsFromDraft(draft)).toEqual(blank);
  });

  it("round-trips every edited field through the URL, commas in checks included", () => {
    const edited: SetupDraftParams = {
      q: "agent:support status:error",
      name: "Refund quality",
      lookback: 72,
      sample: 12.5,
      max: 40,
      context: "Refund only with a receipt",
      watch: ["watch_looping"],
      checks: ["Quotes a price, then changes it"],
      monitor: false,
      every: 60,
    };
    const url = createSerializer(SETUP_DRAFT_PARSERS)(edited);
    const parsed = Object.fromEntries(
      Object.entries(SETUP_DRAFT_PARSERS).map(([key, parser]) => [
        key,
        parser.parseServerSide(new URLSearchParams(url).get(key) ?? undefined),
      ]),
    ) as SetupDraftParams;
    expect(parsed).toEqual(edited);
    expect(paramsFromDraft(draftFromParams(parsed))).toEqual(edited);
  });

  it("drops watch ids it does not know and keeps an explicitly empty watch list", () => {
    expect(draftFromParams({ ...blank, watch: ["watch_unsafe", "watch_gone"] }).watching).toEqual(["watch_unsafe"]);
    const none = draftFromParams({ ...blank, watch: [] });
    expect(none.watching).toEqual([]);
    expect(paramsFromDraft(none).watch).toEqual([]);
  });

  it("leaves half-typed numbers out of the URL instead of writing NaN", () => {
    const draft = draftFromParams(blank);
    const typing = { ...draft, selection: { ...draft.selection, lookback_hours: NaN, sample_percent: NaN } };
    expect(paramsFromDraft(typing)).toEqual(blank);
  });
});
