import { beforeEach, describe, expect, it, vi } from "vitest";
import { fetchAvailableModels } from "./fetch_models";

vi.mock("@/components/networking", () => ({
  apiClient: {
    get: vi.fn(),
  },
}));

import { apiClient } from "@/components/networking";

const mockGet = vi.mocked(apiClient.get);

describe("fetchAvailableModels", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("returns key-scoped models from /v1/models and attaches modes from model_group/info", async () => {
    mockGet.mockImplementation(async (path: string) => {
      if (path === "/model_group/info") {
        return {
          data: [
            { model_group: "anthropic-haiku-4-5", mode: "chat" },
            { model_group: "gpt-realtime", mode: "realtime" },
          ],
        };
      }
      if (path === "/v1/models") {
        return {
          data: [{ id: "anthropic-haiku-4-5" }],
        };
      }
      throw new Error(`unexpected path ${path}`);
    });

    const models = await fetchAvailableModels("sk-virtual-key");

    expect(models).toEqual([{ model_group: "anthropic-haiku-4-5", mode: "chat" }]);
    expect(mockGet).toHaveBeenCalledWith("/v1/models", { accessToken: "sk-virtual-key" });
  });

  it("falls back to model_group/info when /v1/models is empty", async () => {
    mockGet.mockImplementation(async (path: string) => {
      if (path === "/model_group/info") {
        return {
          data: [
            { model_group: "gpt-4o", mode: "chat" },
            { id: "legacy-model", mode: "chat" },
          ],
        };
      }
      if (path === "/v1/models") {
        return { data: [] };
      }
      throw new Error(`unexpected path ${path}`);
    });

    const models = await fetchAvailableModels("sk-admin");
    expect(models.map((m) => m.model_group)).toEqual(["gpt-4o", "legacy-model"]);
  });

  it("still returns /v1/models when model_group/info fails", async () => {
    mockGet.mockImplementation(async (path: string) => {
      if (path === "/model_group/info") {
        throw new Error("forbidden");
      }
      if (path === "/v1/models") {
        return { data: [{ id: "only-from-key" }] };
      }
      throw new Error(`unexpected path ${path}`);
    });

    const models = await fetchAvailableModels("sk-virtual-key");
    expect(models).toEqual([{ model_group: "only-from-key", mode: undefined }]);
  });

  it("throws when both model endpoints fail", async () => {
    mockGet.mockImplementation(async () => {
      throw new Error("network down");
    });

    await expect(fetchAvailableModels("sk-virtual-key")).rejects.toThrow("network down");
  });
});
