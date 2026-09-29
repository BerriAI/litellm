import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import WebSearchInterceptionSettings from "./WebSearchInterceptionSettings";
import { useWebSearchInterceptionSettings } from "@/app/(dashboard)/hooks/webSearchInterceptionSettings/useWebSearchInterceptionSettings";
import { useUpdateWebSearchInterceptionSettings } from "@/app/(dashboard)/hooks/webSearchInterceptionSettings/useUpdateWebSearchInterceptionSettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

vi.mock("@/app/(dashboard)/hooks/webSearchInterceptionSettings/useWebSearchInterceptionSettings", () => ({
  useWebSearchInterceptionSettings: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/webSearchInterceptionSettings/useUpdateWebSearchInterceptionSettings", () => ({
  useUpdateWebSearchInterceptionSettings: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(),
}));

vi.mock("@/components/networking", () => ({
  fetchSearchTools: vi.fn().mockResolvedValue({
    search_tools: [{ search_tool_name: "my-perplexity-search" }, { search_tool_name: "backup-search" }],
  }),
}));

const mockMutate = vi.fn();

const ENABLED_PAYLOAD = {
  enabled: true,
  enabled_providers: ["bedrock"],
  search_tool_name: "my-perplexity-search",
  max_agentic_loops: null,
};

const storedSettings = {
  field_schema: {
    properties: {
      enabled: { description: "Serve web search tool calls from a configured search tool" },
    },
  },
  values: {
    enabled: false,
    enabled_providers: ["bedrock"],
    search_tool_name: "my-perplexity-search",
    max_agentic_loops: null,
  },
};

async function renderSettings() {
  const result = render(<WebSearchInterceptionSettings />);
  await act(async () => {});
  return result;
}

describe("WebSearchInterceptionSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useAuthorized).mockReturnValue({ accessToken: "test-token" } as any);
    vi.mocked(useWebSearchInterceptionSettings).mockReturnValue({
      data: storedSettings,
      isLoading: false,
      isError: false,
      error: null,
    } as any);
    vi.mocked(useUpdateWebSearchInterceptionSettings).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      error: null,
    } as any);
  });

  it("renders the settings section", async () => {
    await renderSettings();
    expect(screen.getByText("Web Search Interception")).toBeInTheDocument();
  });

  it("shows a login prompt when there is no access token", () => {
    vi.mocked(useAuthorized).mockReturnValue({ accessToken: null } as any);
    render(<WebSearchInterceptionSettings />);
    expect(screen.getByText(/please log in/i)).toBeInTheDocument();
  });

  it("hides the settings while loading", async () => {
    vi.mocked(useWebSearchInterceptionSettings).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    } as any);
    await renderSettings();
    expect(screen.queryByText("Enable Web Search Interception")).not.toBeInTheDocument();
  });

  it("surfaces a load failure", async () => {
    vi.mocked(useWebSearchInterceptionSettings).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error("boom"),
    } as any);
    await renderSettings();
    expect(screen.getByText("Could not load web search interception settings")).toBeInTheDocument();
    expect(screen.getByText("boom")).toBeInTheDocument();
  });

  it("keeps save disabled until something changes", async () => {
    const user = userEvent.setup();
    await renderSettings();

    const save = screen.getByRole("button", { name: /save settings/i });
    expect(save).toBeDisabled();

    await user.click(save);
    expect(mockMutate).not.toHaveBeenCalled();
  });

  it("submits the stored values with the toggled enabled flag", async () => {
    const user = userEvent.setup();
    await renderSettings();

    await user.click(screen.getByRole("switch"));
    await user.click(screen.getByRole("button", { name: /save settings/i }));

    expect(mockMutate).toHaveBeenCalledTimes(1);
    expect(mockMutate.mock.calls[0][0]).toEqual(ENABLED_PAYLOAD);
  });

  it("warns when the cluster has it on but the serving pod has not applied it", async () => {
    vi.mocked(useWebSearchInterceptionSettings).mockReturnValue({
      data: { ...storedSettings, values: { ...storedSettings.values, enabled: true }, active_on_this_pod: false },
      isLoading: false,
      isError: false,
      error: null,
    } as any);

    await renderSettings();

    expect(screen.getByText(/has not applied it/i)).toBeInTheDocument();
  });

  it("stays quiet when the serving pod has applied the cluster setting", async () => {
    vi.mocked(useWebSearchInterceptionSettings).mockReturnValue({
      data: { ...storedSettings, values: { ...storedSettings.values, enabled: true }, active_on_this_pod: true },
      isLoading: false,
      isError: false,
      error: null,
    } as any);

    await renderSettings();

    expect(screen.queryByText(/has not applied it/i)).not.toBeInTheDocument();
  });

  it("ignores stored values whose types do not match the field", async () => {
    vi.mocked(useWebSearchInterceptionSettings).mockReturnValue({
      data: {
        ...storedSettings,
        values: {
          enabled: "yes",
          enabled_providers: "bedrock",
          search_tool_name: 7,
          max_agentic_loops: "3",
        },
      },
      isLoading: false,
      isError: false,
      error: null,
    } as any);

    await renderSettings();

    expect(screen.getByRole("switch")).not.toBeChecked();
    expect(screen.getByLabelText(/max agentic loops/i)).toHaveValue(null);
    expect(screen.queryByText("bedrock")).not.toBeInTheDocument();
  });

  it("reseeds the form when the stored settings change underneath it", async () => {
    vi.mocked(useWebSearchInterceptionSettings).mockReturnValue({
      data: { ...storedSettings, values: { ...storedSettings.values, max_agentic_loops: 3 } },
      isLoading: false,
      isError: false,
      error: null,
    } as any);
    const { rerender } = await renderSettings();
    expect(screen.getByRole("spinbutton")).toHaveValue(3);

    vi.mocked(useWebSearchInterceptionSettings).mockReturnValue({
      data: { ...storedSettings, values: { ...storedSettings.values, max_agentic_loops: 9 } },
      isLoading: false,
      isError: false,
      error: null,
    } as any);
    await act(async () => {
      rerender(<WebSearchInterceptionSettings />);
    });

    expect(screen.getByRole("spinbutton")).toHaveValue(9);
  });

  it("sends null rather than a number when the loop cap is cleared", async () => {
    const user = userEvent.setup();
    vi.mocked(useWebSearchInterceptionSettings).mockReturnValue({
      data: { ...storedSettings, values: { ...storedSettings.values, max_agentic_loops: 5 } },
      isLoading: false,
      isError: false,
      error: null,
    } as any);
    await renderSettings();

    await user.clear(screen.getByRole("spinbutton"));
    await user.click(screen.getByRole("button", { name: /save settings/i }));

    expect(mockMutate).toHaveBeenCalledTimes(1);
    expect(mockMutate.mock.calls[0][0].max_agentic_loops).toBeNull();
  });
});
