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
    expect(modelAvailableCallMock).not.toHaveBeenCalled();
  });

  it.each([
    ["an error payload in place of the list", { data: { error: "no access" } }],
    ["a missing data key", {}],
    ["no body at all", undefined],
  ])("falls back to /models on %s so non-admin sessions still get a dropdown", async (_label, hubResponse) => {
    modelHubCallMock.mockResolvedValue(hubResponse);
    modelAvailableCallMock.mockResolvedValue({
      data: [
        { id: "llama", mode: "chat" },
        { id: "gpt-4o", mode: "chat" },
      ],
    });

    expect(await fetchAvailableModels("token")).toEqual([
      { model_group: "gpt-4o", mode: "chat" },
      { model_group: "llama", mode: "chat" },
    ]);
    expect(modelAvailableCallMock).toHaveBeenCalledWith("token", "", "");
  });

  it("falls back to /models when the model hub call fails", async () => {
    modelHubCallMock.mockRejectedValue(new Error("network down"));
    modelAvailableCallMock.mockResolvedValue({ data: [{ id: "gpt-4o", mode: "chat" }] });

    expect(await fetchAvailableModels("token")).toEqual([{ model_group: "gpt-4o", mode: "chat" }]);
  });

  it("deduplicates the fallback list by model group", async () => {
    modelHubCallMock.mockResolvedValue({ data: [] });
    modelAvailableCallMock.mockResolvedValue({ data: [{ id: "gpt-4o" }, { id: "gpt-4o" }] });

    expect(await fetchAvailableModels("token")).toEqual([{ model_group: "gpt-4o" }]);
  });

  it("excludes the all-proxy-models permission sentinel from fallback options", async () => {
    modelHubCallMock.mockResolvedValue({ data: [] });
    modelAvailableCallMock.mockResolvedValue({
      data: [{ id: "all-proxy-models" }, { id: "gpt-4o", mode: "chat" }],
    });

    expect(await fetchAvailableModels("token")).toEqual([{ model_group: "gpt-4o", mode: "chat" }]);
  });

  it("returns an empty list when both routes come back empty", async () => {
    modelHubCallMock.mockResolvedValue({ data: [] });
    modelAvailableCallMock.mockResolvedValue({ data: [] });

    expect(await fetchAvailableModels("token")).toEqual([]);
  });
});
