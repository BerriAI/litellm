import { beforeEach, describe, expect, it, vi } from "vitest";
import { modelAvailableCall, modelHubCall } from "@/components/networking";
import { fetchAvailableModels, fetchAvailableModelsForTeam } from "./fetch_models";

vi.mock("@/components/networking", () => ({
  modelAvailableCall: vi.fn(),
  modelHubCall: vi.fn(),
}));

const modelAvailableCallMock = vi.mocked(modelAvailableCall);
const modelHubCallMock = vi.mocked(modelHubCall);

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

describe("fetchAvailableModels", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("carries the reasoning capabilities the model hub reports for each group", async () => {
    modelHubCallMock.mockResolvedValue({
      data: [
        { model_group: "smart", mode: "chat", supports_reasoning: true, supported_reasoning_efforts: ["low", "high"] },
        { model_group: "plain", mode: "chat", supports_reasoning: false },
      ],
    });

    expect(await fetchAvailableModels("token")).toEqual([
      { model_group: "plain", mode: "chat" },
      { model_group: "smart", mode: "chat", supports_reasoning: true, supported_reasoning_efforts: ["low", "high"] },
    ]);
  });

  it.each([
    ["an error payload in place of the list", { data: { error: "no access" } }],
    ["a missing data key", {}],
    ["no body at all", undefined],
  ])("returns an empty list on %s rather than throwing", async (_label, response) => {
    modelHubCallMock.mockResolvedValue(response);

    expect(await fetchAvailableModels("token")).toEqual([]);
  });
});
