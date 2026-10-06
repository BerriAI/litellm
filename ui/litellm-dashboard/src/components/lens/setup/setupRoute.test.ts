import { createLoader, createSerializer } from "nuqs";
import { describe, expect, it } from "vitest";
import { investigationDefaults } from "./investigationSchema";
import { draftFromParams, paramsFromDraft, SETUP_DRAFT_PARSERS } from "./setupRoute";

const toUrl = createSerializer(SETUP_DRAFT_PARSERS);
const fromUrl = createLoader(SETUP_DRAFT_PARSERS);
const blank = investigationDefaults(undefined, "new", "traces");

describe("setup draft params", () => {
  it("opens a blank draft from an empty URL and keeps an untouched draft out of it", () => {
    expect(draftFromParams(fromUrl(""))).toEqual(blank);
    expect(toUrl(paramsFromDraft(blank))).toBe("");
  });

  it("round-trips every edited field through the URL, commas in checks included", () => {
    const edited = {
      ...blank,
      name: "Refund quality",
      selection: {
        ...blank.selection,
        source: "both" as const,
        agent_name: "support",
        service: "billing-api",
        filters: [
          { key: "tier", value: "gold=vip" },
          { key: "region", value: "eu,us" },
        ],
        team_id: "team-1",
        lookback_hours: 72,
        sample_percent: 12.5,
        sample_size: 40,
      },
      context: "Refund only with a receipt",
      watching: ["watch_looping"],
      questions: [{ id: "c1", instruction: "Quotes a price, then changes it", enabled: true }],
      repeat: !blank.repeat,
      interval: 60,
      selectedModel: "gpt-4o-mini",
      budget: 25.5,
    };
    const restored = draftFromParams(fromUrl(toUrl(paramsFromDraft(edited))));
    expect({ ...restored, questions: restored.questions.map((check) => check.instruction) }).toEqual({
      ...edited,
      questions: ["Quotes a price, then changes it"],
    });
  });

  it("drops watch ids it does not know and keeps an explicitly empty watch list", () => {
    expect(draftFromParams(fromUrl("?watch=watch_unsafe,watch_gone")).watching).toEqual(["watch_unsafe"]);
    const none = draftFromParams(fromUrl(toUrl(paramsFromDraft({ ...blank, watching: [] }))));
    expect(none.watching).toEqual([]);
  });

  it("keeps the tracing default out of the URL and restores it for a draft without a source", () => {
    const requests = investigationDefaults(undefined, "new", "requests");
    expect(toUrl(paramsFromDraft(requests, "requests"))).toBe("");
    expect(draftFromParams(fromUrl(""), "requests").selection.source).toBe("requests");
    expect(toUrl(paramsFromDraft(blank, "requests"))).toBe("?source=traces");
  });

  it("leaves half-typed numbers and blank checks out of the URL", () => {
    const typing = {
      ...blank,
      selection: { ...blank.selection, lookback_hours: NaN, sample_percent: NaN },
      questions: [{ id: "c1", instruction: "  ", enabled: true }],
    };
    expect(toUrl(paramsFromDraft(typing))).toBe("");
  });
});
