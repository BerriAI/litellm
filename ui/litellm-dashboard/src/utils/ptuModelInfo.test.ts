import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import { describe, expect, it } from "vitest";
import { applyPtuModelInfo, PTU_MODEL_INFO_FIELDS } from "./ptuModelInfo";

dayjs.extend(utc);

const storedModelInfo = () => ({
  id: "model-1",
  team_id: "team-1",
  ptu_count: 15,
  cost_per_ptu_per_hour: 2,
  ptu_effective_from: "2026-07-01T00:00:00.000Z",
  ptu_effective_to: "2026-08-01T00:00:00.000Z",
});

describe("applyPtuModelInfo", () => {
  it("folds the form values into model_info when PTU cost attribution is enabled", () => {
    const result = applyPtuModelInfo(
      { id: "model-1", team_id: "team-1" },
      {
        ptu_count: "20",
        cost_per_ptu_per_hour: "3.5",
        ptu_effective_from: dayjs.utc("2026-09-01T00:00:00.000Z"),
        ptu_effective_to: null,
      },
      true,
    );

    expect(result).toEqual({
      id: "model-1",
      team_id: "team-1",
      ptu_count: 20,
      cost_per_ptu_per_hour: 3.5,
      ptu_effective_from: "2026-09-01T00:00:00.000Z",
      ptu_effective_to: null,
    });
  });

  it("sends an explicit null for a field the operator cleared while enabled", () => {
    const result = applyPtuModelInfo(storedModelInfo(), { ptu_count: "", cost_per_ptu_per_hour: "" }, true);

    expect(result.ptu_count).toBeNull();
    expect(result.cost_per_ptu_per_hour).toBeNull();
  });

  it("strips every PTU field from the payload when PTU cost attribution is disabled", () => {
    const result = applyPtuModelInfo(storedModelInfo(), { ptu_count: "20", cost_per_ptu_per_hour: "3.5" }, false);

    for (const field of PTU_MODEL_INFO_FIELDS) {
      expect(Object.keys(result)).not.toContain(field);
    }
    expect(result).toEqual({ id: "model-1", team_id: "team-1" });
  });

  it("never sends a null PTU field when disabled, so an unrelated save cannot clear stored config", () => {
    const result = applyPtuModelInfo(storedModelInfo(), {}, false);

    expect(Object.values(result)).not.toContain(null);
    expect("ptu_count" in result).toBe(false);
  });

  it("leaves non-PTU model_info untouched when disabled", () => {
    const result = applyPtuModelInfo({ id: "model-1", access_groups: ["a"], health_check_model: "gpt-5.2" }, {}, false);

    expect(result).toEqual({ id: "model-1", access_groups: ["a"], health_check_model: "gpt-5.2" });
  });
});
