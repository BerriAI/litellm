import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { createElement, type ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getRouterSettingsCall, setCallbacksCall } from "@/components/networking";
import type { RoutingGroup } from "@/components/routing_groups/types";
import { useRoutingGroups, useSaveRoutingGroups } from "./useRoutingGroups";

vi.mock("@/components/networking", () => ({
  getRouterSettingsCall: vi.fn(),
  setCallbacksCall: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "test-token", userId: "admin", userRole: "Admin" }),
}));

const createWrapper = () => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return function QueryWrapper({ children }: { children: ReactNode }) {
    return createElement(QueryClientProvider, { client }, children);
  };
};

describe("routing group settings", () => {
  beforeEach(() => vi.clearAllMocks());

  it.each([
    { metadata: ["simple-shuffle", "priority"], expected: ["simple-shuffle", "priority"] },
    { metadata: undefined, expected: ["simple-shuffle"] },
  ])("uses the advertised group strategies and supports older gateways", async ({ metadata, expected }) => {
    vi.mocked(getRouterSettingsCall).mockResolvedValue({
      fields: [{ field_name: "routing_strategy", options: ["simple-shuffle"] }],
      routing_group_strategies: metadata,
      current_values: {},
    });
    const { result } = renderHook(() => useRoutingGroups(), { wrapper: createWrapper() });

    await waitFor(() => expect(result.current.data?.availableStrategies).toEqual(expected));
  });

  it("loads and saves membership priorities through the existing router settings endpoint", async () => {
    const groups: RoutingGroup[] = [
      {
        group_name: "preferred-chat",
        models: ["preferred", "backup"],
        routing_strategy: "priority",
        model_priorities: { preferred: 1, backup: 4 },
      },
    ];
    vi.mocked(getRouterSettingsCall).mockResolvedValue({ current_values: { routing_groups: groups } });
    vi.mocked(setCallbacksCall).mockResolvedValue({});
    const { result } = renderHook(() => ({ query: useRoutingGroups(), save: useSaveRoutingGroups() }), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.query.data?.routingGroups).toEqual(groups));
    await act(async () => {
      await result.current.save.mutateAsync(groups);
    });

    expect(setCallbacksCall).toHaveBeenCalledWith("test-token", { router_settings: { routing_groups: groups } });
  });
});
