/* @vitest-environment jsdom */
import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useGetKeyModels } from "./useGetKeyModels";
import * as networking from "@/components/networking";

vi.mock("@/components/networking", () => ({
  fetchKeyModelCall: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(() => ({
    accessToken: "test-token-456",
  })),
}));

const emptyKeyModelResponse = {
  model_display_sections: [],
  source: "no-default-models",
  resolved_total_count: 0,
  matched_count: 0,
  models_truncated: false,
  all_team_models_without_team: false,
};

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

const wrapper = ({ children }: { children: React.ReactNode }) => {
  const queryClient = createQueryClient();
  return React.createElement(QueryClientProvider, { client: queryClient }, children);
};

describe("useGetKeyModels", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    const useAuthorizedModule = await import("@/app/(dashboard)/hooks/useAuthorized");
    vi.mocked(useAuthorizedModule.default).mockReturnValue({
      accessToken: "test-token-456",
    } as any);
  });

  it("should load default full model list without compact", async () => {
    vi.mocked(networking.fetchKeyModelCall).mockResolvedValue({
      ...emptyKeyModelResponse,
      resolved_total_count: 3,
      model_display_sections: [
        { title: "Other models", section_kind: "ungrouped", models: ["a", "b", "c"] },
      ],
    });

    const { result } = renderHook(() => useGetKeyModels("test-key-id"), { wrapper });

    expect(result.current).toHaveProperty("defaultModelsQuery");
    expect(result.current).toHaveProperty("searchQuery");

    await waitFor(() => expect(result.current.defaultModelsQuery.isSuccess).toBe(true));
    expect(networking.fetchKeyModelCall).toHaveBeenCalledWith("test-token-456", "test-key-id");
    expect(result.current.defaultModelsQuery.data?.resolved_total_count).toBe(3);
  });
});
