import { describe, expect, it } from "vitest";
import { activityLink, presentAlert, type SavedAlert } from "./alertPresentation";

const saved: SavedAlert = {
  id: "1",
  type: "spend",
  resource_type: "team",
  resource_id: "example-team",
  status: "open",
  window_start: "2026-01-08T00:00:00Z",
  window_end: "2026-01-09T00:00:00Z",
  created_at: "2026-01-09T03:00:00Z",
  updated_at: "2026-01-09T03:00:00Z",
  evidence: { value: 20, baseline: 2, unit: "usd", sample_size: 100 },
};

describe("alert presentation", () => {
  it.each(
    (
      [
        ["spend", "usd", 20, 2, "Spend", "$20.00"],
        ["traffic", "requests", 1000, 100, "Traffic", "1,000"],
        ["latency", "milliseconds", 4000, 1000, "Mean response time", "4 s"],
        ["reliability", "ratio", 0.2, 0.01, "Error rate", "20%"],
      ] as const
    ).map((row) => [row] as const),
  )("labels %s with the recorded units", ([type, unit, value, baseline, label, formatted]) => {
    const alert = { ...saved, type, evidence: { value, baseline, unit, sample_size: 100 } };
    const presentation = presentAlert({ source: "performance", alert });
    expect(presentation.title).toContain(`${label} on example-team`);
    expect(presentation.description).toContain(formatted);
    expect(presentation.baseline).toBe(baseline);
  });
  it("links to the resource with an inclusive eight-day window", () => {
    expect(activityLink(saved)).toBe(
      "/ui/teams?team=example-team&detail_tab=activity&activity_from=2026-01-01&activity_to=2026-01-08",
    );
    expect(activityLink({ ...saved, resource_type: "user", resource_id: "reader@example.com" })).toContain(
      "user=reader%40example.com&detail_tab=activity",
    );
    expect(activityLink({ ...saved, resource_type: "gateway", resource_id: "gateway" })).toBe(
      "/ui/usage?from=2026-01-01&to=2026-01-08",
    );
  });
  it("links model groups to routing instead of an unrelated deployment's Activity", () => {
    const presentation = presentAlert({
      source: "performance",
      alert: { ...saved, resource_type: "model", resource_id: "research/model" },
    });
    expect(presentation.href).toBe("/ui/models-and-endpoints?model_group=research%2Fmodel");
    expect(presentation.linkLabel).toBe("View model routing");
  });
  it("keeps the saved comparison when current Usage has changed", () => {
    const presentation = presentAlert({
      source: "performance",
      alert: { ...saved, resource_name: "Research", history: [{ date: "2026-01-08", value: 40 }] },
    });
    expect(presentation.description).toContain("$20.00");
    expect(presentation.history?.[0].value).toBe(40);
    expect(presentation.title).toContain("Research");
  });
  it("shows retirement and last-used dates without inventing a detection date", () => {
    const presentation = presentAlert({
      source: "deprecation",
      alert: {
        id: "retirement",
        usageModel: "example/retiring-model",
        requests: 20,
        lastDayRequests: 20,
        lastUsedDate: "2026-01-08",
        model: {
          model_name: "example-group",
          deprecation_date: "2026-01-31",
          days_until_deprecation: 23,
          status: "imminent",
        },
      },
    });
    expect(presentation.dates).toEqual([
      ["Retires", "2026-01-31"],
      ["Last used", "2026-01-08"],
    ]);
    expect(presentation.status).toBe("open");
  });
});
