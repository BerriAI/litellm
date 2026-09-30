import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, act, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import MCPSemanticFilterSettings from "./MCPSemanticFilterSettings";
import { useMCPSemanticFilterSettings } from "@/app/(dashboard)/hooks/mcpSemanticFilterSettings/useMCPSemanticFilterSettings";
import { useUpdateMCPSemanticFilterSettings } from "@/app/(dashboard)/hooks/mcpSemanticFilterSettings/useUpdateMCPSemanticFilterSettings";
import { fetchAvailableModels } from "@/components/llm_calls/fetch_models";

vi.mock("@/app/(dashboard)/hooks/mcpSemanticFilterSettings/useMCPSemanticFilterSettings", () => ({
  useMCPSemanticFilterSettings: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/mcpSemanticFilterSettings/useUpdateMCPSemanticFilterSettings", () => ({
  useUpdateMCPSemanticFilterSettings: vi.fn(),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
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

const AVAILABLE_MODELS = [
  { model_group: "text-embedding-3-large", mode: "embedding" },
  { model_group: "gpt-4o", mode: "chat" },
];

const SWITCH_ONLY_PAYLOAD = {
  enabled: true,
  embedding_model: "text-embedding-3-small",
  top_k: 10,
  similarity_threshold: 0.3,
};

const DEFAULTED_PAYLOAD = {
  enabled: false,
  embedding_model: "text-embedding-3-small",
  top_k: 7,
  similarity_threshold: 0.3,
};

const FULLY_EDITED_PAYLOAD = {
  enabled: true,
  embedding_model: "text-embedding-3-large",
  top_k: 25,
  similarity_threshold: 0.35,
};

const CLEARED_TOP_K_PAYLOAD = {
  enabled: false,
  embedding_model: "text-embedding-3-small",
  top_k: null,
  similarity_threshold: 0.3,
};

const CLAMPED_MAX_PAYLOAD = {
  enabled: false,
  embedding_model: "text-embedding-3-small",
  top_k: 100,
  similarity_threshold: 0.3,
};

const CLAMPED_MIN_PAYLOAD = {
  enabled: false,
  embedding_model: "text-embedding-3-small",
  top_k: 1,
  similarity_threshold: 0.3,
};

// Helper that renders the component and flushes the fetchAvailableModels effect
async function renderSettings(props: React.ComponentProps<typeof MCPSemanticFilterSettings>) {
  const result = render(<MCPSemanticFilterSettings {...props} />);
  if (props.accessToken) {
    // Let the async fetchAvailableModels effect settle to avoid act() warnings
    await act(async () => {});
  }
  return result;
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
    expect(screen.getByText("Could not load MCP Semantic Filter settings")).toBeInTheDocument();
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
    expect(screen.getByRole("button", { name: /save settings/i })).toBeDisabled();
  });

  it("should enable Save Settings button after a form field is changed", async () => {
    const user = userEvent.setup();
    await renderSettings({ accessToken: "test-token" });

    expect(screen.getByRole("button", { name: /save settings/i })).toBeDisabled();

    await user.click(screen.getByRole("switch"));

    expect(screen.getByRole("button", { name: /save settings/i })).toBeEnabled();
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

  it("should send the default values as the payload when only the switch is toggled", async () => {
    const user = userEvent.setup();
    await renderSettings({ accessToken: "test-token" });

    await user.click(screen.getByRole("switch"));
    await user.click(screen.getByRole("button", { name: /save settings/i }));

    expect(mockMutate).toHaveBeenCalledWith(SWITCH_ONLY_PAYLOAD, expect.anything());
  });

  it("should fall back to the hardcoded defaults when the backend returns no values", async () => {
    vi.mocked(useMCPSemanticFilterSettings).mockReturnValue({
      data: { field_schema: defaultSettingsData.field_schema, values: {} },
      isLoading: false,
      isError: false,
      error: null,
    } as any);
    const user = userEvent.setup();
    await renderSettings({ accessToken: "test-token" });

    const topK = screen.getByRole("spinbutton");
    await user.clear(topK);
    fireEvent.change(topK, { target: { value: "7" } });
    await user.click(screen.getByRole("button", { name: /save settings/i }));

    expect(mockMutate).toHaveBeenCalledWith(DEFAULTED_PAYLOAD, expect.anything());
  });

  it("should send every edited field in the payload", async () => {
    vi.mocked(fetchAvailableModels).mockResolvedValueOnce(AVAILABLE_MODELS);
    const user = userEvent.setup();
    await renderSettings({ accessToken: "test-token" });

    await user.click(screen.getByRole("switch"));

    const topK = screen.getByRole("spinbutton");
    await user.clear(topK);
    fireEvent.change(topK, { target: { value: "25" } });

    const slider = screen.getByRole("slider", { hidden: true });
    fireEvent.keyDown(slider, { key: "ArrowRight", keyCode: 39, which: 39 });

    const embeddingModel = screen.getByRole("combobox");
    await user.click(embeddingModel);
    await user.clear(embeddingModel);
    fireEvent.change(embeddingModel, { target: { value: "large" } });
    fireEvent.keyDown(embeddingModel, { key: "ArrowDown", keyCode: 40, which: 40 });
    fireEvent.keyDown(embeddingModel, { key: "Enter", keyCode: 13, which: 13 });

    await user.click(screen.getByRole("button", { name: /save settings/i }));

    expect(mockMutate).toHaveBeenCalledWith(FULLY_EDITED_PAYLOAD, expect.anything());
  });

  it("should send top_k as null when the field is cleared", async () => {
    const user = userEvent.setup();
    await renderSettings({ accessToken: "test-token" });

    const topK = screen.getByRole("spinbutton");
    await user.clear(topK);
    await user.click(screen.getByRole("button", { name: /save settings/i }));

    expect(mockMutate).toHaveBeenCalledWith(CLEARED_TOP_K_PAYLOAD, expect.anything());
  });

  it("should clamp top_k above the maximum back into range on blur", async () => {
    const user = userEvent.setup();
    await renderSettings({ accessToken: "test-token" });

    const topK = screen.getByRole("spinbutton");
    await user.clear(topK);
    fireEvent.change(topK, { target: { value: "500" } });
    await user.tab();
    await user.click(screen.getByRole("button", { name: /save settings/i }));

    expect(mockMutate).toHaveBeenCalledWith(CLAMPED_MAX_PAYLOAD, expect.anything());
  });

  it("should clamp top_k below the minimum back into range on blur", async () => {
    const user = userEvent.setup();
    await renderSettings({ accessToken: "test-token" });

    const topK = screen.getByRole("spinbutton");
    await user.clear(topK);
    fireEvent.change(topK, { target: { value: "0" } });
    await user.tab();
    await user.click(screen.getByRole("button", { name: /save settings/i }));

    expect(mockMutate).toHaveBeenCalledWith(CLAMPED_MIN_PAYLOAD, expect.anything());
  });
  it("offers no way to clear the embedding model, as the antd Select had no allowClear", async () => {
    const { container } = await renderSettings({ accessToken: "test-token" });

    expect(screen.getByRole("combobox")).toHaveValue("text-embedding-3-small");
    expect(container.querySelector('[data-slot="combobox-clear"]')).not.toBeInTheDocument();
  });
});
