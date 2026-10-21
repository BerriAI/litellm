import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import MCPToolsListCacheSettings from "./MCPToolsListCacheSettings";
import {
  useMCPToolsListCacheSettings,
  useUpdateMCPToolsListCacheSettings,
} from "@/app/(dashboard)/hooks/mcpToolsListCacheSettings/useMCPToolsListCacheSettings";

vi.mock("@/app/(dashboard)/hooks/mcpToolsListCacheSettings/useMCPToolsListCacheSettings", () => ({
  useMCPToolsListCacheSettings: vi.fn(),
  useUpdateMCPToolsListCacheSettings: vi.fn(),
}));

const mockMutate = vi.fn();
const storedSettings = { field_schema: {}, values: { ttl_seconds: 0 } };

describe("MCPToolsListCacheSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useMCPToolsListCacheSettings).mockReturnValue({
      data: storedSettings,
      isLoading: false,
      isError: false,
      error: null,
    } as ReturnType<typeof useMCPToolsListCacheSettings>);
    vi.mocked(useUpdateMCPToolsListCacheSettings).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
    } as unknown as ReturnType<typeof useUpdateMCPToolsListCacheSettings>);
  });

  it("saves the configured TTL in seconds", () => {
    render(<MCPToolsListCacheSettings accessToken="token" />);
    fireEvent.change(screen.getByLabelText("Cache TTL in seconds"), { target: { value: "30" } });
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }));

    expect(mockMutate).toHaveBeenCalledTimes(1);
    expect(mockMutate.mock.calls[0][0]).toEqual({ ttl_seconds: 30 });
  });
});
