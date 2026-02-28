import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import MCPSemanticFilterSettings from "./MCPSemanticFilterSettings";
import { useMCPSemanticFilterSettings } from "@/app/(dashboard)/hooks/mcpSemanticFilterSettings/useMCPSemanticFilterSettings";
import { useUpdateMCPSemanticFilterSettings } from "@/app/(dashboard)/hooks/mcpSemanticFilterSettings/useUpdateMCPSemanticFilterSettings";

vi.mock(
  "@/app/(dashboard)/hooks/mcpSemanticFilterSettings/useMCPSemanticFilterSettings",
  () => ({ useMCPSemanticFilterSettings: vi.fn() })
);

vi.mock(
  "@/app/(dashboard)/hooks/mcpSemanticFilterSettings/useUpdateMCPSemanticFilterSettings",
  () => ({ useUpdateMCPSemanticFilterSettings: vi.fn() })
);

vi.mock("@/components/playground/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([]),
}));

vi.mock("./MCPSemanticFilterTestPanel", () => ({
  default: () => <div data-testid="mcp-test-panel" />,
}));

vi.mock("./semanticFilterTestUtils", () => ({
  getCurlCommand: vi.fn().mockReturnValue("curl ..."),
  runSemanticFilterTest: vi.fn(),
}));

const mockMutate = vi.fn();

const defaultSettingsData = {
  field_schema: {
    properties: {
      enabled: { description: "Enable semantic filtering for MCP tools" },
    },
  },
  values: {
    enabled: false,
    embedding_model: "text-embedding-3-small",
    top_k: 10,
    similarity_threshold: 0.3,
  },
};

// Helper that renders the component and flushes the fetchAvailableModels effect
async function renderSettings(props: React.ComponentProps<typeof MCPSemanticFilterSettings>) {
  render(<MCPSemanticFilterSettings {...props} />);
  if (props.accessToken) {
    // Let the async fetchAvailableModels effect settle to avoid act() warnings
    await act(async () => {});
  }
}

describe("MCPSemanticFilterSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(useMCPSemanticFilterSettings).mockReturnValue({
      data: defaultSettingsData,
      isLoading: false,
      isError: false,
      error: null,
    } as any);
    vi.mocked(useUpdateMCPSemanticFilterSettings).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      error: null,
    } as any);
  });

  it("should render", async () => {
    await renderSettings({ accessToken: "test-token" });
    expect(screen.getByText("Semantic Tool Filtering")).toBeInTheDocument();
  });

  it("should show a login prompt when accessToken is null", () => {
    render(<MCPSemanticFilterSettings accessToken={null} />);
    expect(screen.getByText(/please log in/i)).toBeInTheDocument();
  });

  it("should not render the form when accessToken is null", () => {
    render(<MCPSemanticFilterSettings accessToken={null} />);
    expect(screen.queryByText("Enable Semantic Filtering")).not.toBeInTheDocument();
  });

  it("should not show the settings content while loading", async () => {
    vi.mocked(useMCPSemanticFilterSettings).mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false,
      error: null,
    } as any);
    await renderSettings({ accessToken: "test-token" });
    expect(screen.queryByText("Semantic Tool Filtering")).not.toBeInTheDocument();
  });

  it("should show an error alert when data fails to load", async () => {
    vi.mocked(useMCPSemanticFilterSettings).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error("Network error"),
    } as any);
    await renderSettings({ accessToken: "test-token" });
    expect(
      screen.getByText("Could not load MCP Semantic Filter settings")
    ).toBeInTheDocument();
    expect(screen.getByText("Network error")).toBeInTheDocument();
  });

  it("should show the error message from the error object when loading fails", async () => {
    vi.mocked(useMCPSemanticFilterSettings).mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true,
      error: new Error("Connection refused"),
    } as any);
    await renderSettings({ accessToken: "test-token" });
    expect(screen.getByText("Connection refused")).toBeInTheDocument();
  });

  it("should render the info alert and form fields when data is loaded", async () => {
    await renderSettings({ accessToken: "test-token" });
    expect(screen.getByText("Semantic Tool Filtering")).toBeInTheDocument();
    expect(screen.getByText("Enable Semantic Filtering")).toBeInTheDocument();
    expect(screen.getByText("Top K Results")).toBeInTheDocument();
    expect(screen.getByText("Similarity Threshold")).toBeInTheDocument();
  });

  it("should render the test panel", async () => {
    await renderSettings({ accessToken: "test-token" });
    expect(screen.getByTestId("mcp-test-panel")).toBeInTheDocument();
  });

  it("should have Save Settings button disabled initially", async () => {
    await renderSettings({ accessToken: "test-token" });
    expect(
      screen.getByRole("button", { name: /save settings/i })
    ).toBeDisabled();
  });

  it("should enable Save Settings button after a form field is changed", async () => {
    const user = userEvent.setup();
    await renderSettings({ accessToken: "test-token" });

    expect(screen.getByRole("button", { name: /save settings/i })).toBeDisabled();

    await user.click(screen.getByRole("switch"));

    expect(screen.getByRole("button", { name: /save settings/i })).not.toBeDisabled();
  });

  it("should show an error alert when the mutation fails", async () => {
    vi.mocked(useUpdateMCPSemanticFilterSettings).mockReturnValue({
      mutate: mockMutate,
      isPending: false,
      error: new Error("Failed to update settings"),
    } as any);
    await renderSettings({ accessToken: "test-token" });
    expect(screen.getByText("Could not update settings")).toBeInTheDocument();
    expect(screen.getByText("Failed to update settings")).toBeInTheDocument();
  });
});
