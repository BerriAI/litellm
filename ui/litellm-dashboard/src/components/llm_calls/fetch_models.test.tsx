import { beforeEach, describe, expect, it, vi } from "vitest";
import { modelAvailableCall, modelHubCall } from "@/components/networking";
import { fetchAvailableModelsForTeam } from "./fetch_models";

vi.mock("@/components/networking", () => ({
  modelAvailableCall: vi.fn(),
  modelHubCall: vi.fn(),
}));

const modelAvailableCallMock = vi.mocked(modelAvailableCall);

describe("fetchAvailableModelsForTeam", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("requests the models scoped to the team so team-only BYOK models are included", async () => {
    modelAvailableCallMock.mockResolvedValue({
      data: [{ id: "all-proxy-models" }, { id: "openai/*" }, { id: "gpt-5-mini" }, { id: "openai/*" }],
    });

    const models = await fetchAvailableModelsForTeam("token", "team-123");

    expect(modelAvailableCallMock).toHaveBeenCalledWith("token", "", "", false, "team-123");
    expect(models).toEqual([{ model_group: "gpt-5-mini" }, { model_group: "openai/*" }]);
  });

  it("returns an empty list when the team has no models", async () => {
    modelAvailableCallMock.mockResolvedValue({ data: [] });

    expect(await fetchAvailableModelsForTeam("token", "team-123")).toEqual([]);
  });
});

describe("fetchAvailableModelsForTeam capability join", () => {
  it("carries model-group capabilities onto team-allowed names that have a group entry", async () => {
    modelAvailableCallMock.mockResolvedValue({ data: [{ id: "gpt-5-mini" }, { id: "team-only-byok" }] });
    vi.mocked(modelHubCall).mockResolvedValue({
      data: [
        {
          model_group: "gpt-5-mini",
          mode: "chat",
          supports_reasoning: true,
          supported_reasoning_efforts: ["minimal", "low", "medium", "high"],
        },
      ],
    });

    const models = await fetchAvailableModelsForTeam("token", "team-123");

    expect(models).toEqual([
      {
        model_group: "gpt-5-mini",
        mode: "chat",
        supports_reasoning: true,
        supported_reasoning_efforts: ["minimal", "low", "medium", "high"],
      },
      { model_group: "team-only-byok" },
    ]);
  });

  it("keeps the team list usable when the group-info fetch fails", async () => {
    modelAvailableCallMock.mockResolvedValue({ data: [{ id: "gpt-5-mini" }] });
    vi.mocked(modelHubCall).mockRejectedValue(new Error("boom"));

    expect(await fetchAvailableModelsForTeam("token", "team-123")).toEqual([{ model_group: "gpt-5-mini" }]);
  });
});
