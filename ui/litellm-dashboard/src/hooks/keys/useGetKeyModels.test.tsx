/* @vitest-environment jsdom */
import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { UseGetKeyModels } from "./useGetKeyModels";
import * as networking from "@/components/networking";

vi.mock("@/components/networking", () => ({
  fetchKeyModelCall: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(() => ({
    accessToken: "test-token-456",
  })),
}));

const createQueryClient = () =>
  new QueryClient({
  });

const wrapper = ({ children }: { children: React.ReactNode }) => {
  const queryClient = createQueryClient();
  return React.createElement(QueryClientProvider, { client: queryClient }, children);
};

const mockAccessToken = 'test-key-id';
const mockAccessGroups = ["group-1", "group-2", "group-3"];

describe("useGetKeyModels", () => {
  beforeEach(async () => {
    vi.clearAllMocks();
    const useAuthorizedModule = await import("@/app/(dashboard)/hooks/useAuthorized");
    vi.mocked(useAuthorizedModule.default).mockReturnValue({
      accessToken: mockAccessToken,
    } as any);
  });

  it("should return hook result without errors", () => {
    vi.mocked(networking.fetchKeyModelCall).mockResolvedValue({source: '', models: []});

    const { result } = renderHook(() => UseGetKeyModels('test-key-id'), { wrapper });

    expect(result.current).toBeDefined();
    expect(result.current).toHaveProperty("data");
    expect(result.current).toHaveProperty("isSuccess");
    expect(result.current).toHaveProperty("isError");
    expect(result.current).toHaveProperty("status");
  });
});