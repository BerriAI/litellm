import { fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { chooseSelectOption, renderWithProviders as render } from "@/../tests/test-utils";
import CompareUI, { type ComparisonInstance } from "./CompareUI";
import { makeOpenAIChatCompletionRequest } from "@/components/llm_calls/chat_completion";

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([{ model_group: "gpt-4" }, { model_group: "gpt-3.5-turbo" }]),
}));

vi.mock("../../llm_calls/fetch_agents", () => ({
  fetchAvailableAgents: vi.fn().mockResolvedValue([]),
}));

vi.mock("@/components/llm_calls/chat_completion", () => ({
  makeOpenAIChatCompletionRequest: vi.fn().mockResolvedValue(undefined),
}));

let capturedOnImageUpload: ((file: File) => false) | null = null;

vi.mock("../chat_ui/ChatImageUpload", () => ({
  default: ({ onImageUpload }: { onImageUpload: (file: File) => false }) => {
    capturedOnImageUpload = onImageUpload;
    return (
      <div data-testid="chat-image-upload">
        <button data-testid="trigger-upload">Upload</button>
      </div>
    );
  },
}));

vi.mock("../chat_ui/ChatImageUtils", () => ({
  createChatMultimodalMessage: vi.fn().mockResolvedValue({
    role: "user",
    content: [
      { type: "text", text: "test message" },
      { type: "image_url", image_url: { url: "data:image/png;base64,test" } },
    ],
  }),
  createChatDisplayMessage: vi.fn().mockReturnValue({
    role: "user",
    content: "test message [Image attached]",
    imagePreviewUrl: "blob:test-url",
  }),
}));

vi.mock("./components/ComparisonPanel", () => ({
  ComparisonPanel: ({
    comparison,
    onRemove,
    onUpdate,
  }: {
    comparison: ComparisonInstance;
    onRemove: () => void;
    onUpdate: (updates: Partial<ComparisonInstance>) => void;
  }) => (
    <div data-testid={`comparison-panel-${comparison.id}`} data-model={comparison.model}>
      <button data-testid={`remove-${comparison.id}`} onClick={onRemove}>
        Remove
      </button>
      <button data-testid={`pick-${comparison.id}`} onClick={() => onUpdate({ model: "gpt-3.5-turbo" })}>
        Pick
      </button>
    </div>
  ),
}));

vi.mock("./components/MessageInput", () => ({
  MessageInput: ({ value, onChange, onSend, disabled, hasAttachment, uploadComponent }: any) => (
    <div data-testid="message-input">
      {uploadComponent && <div data-testid="upload-component">{uploadComponent}</div>}
      <textarea
        data-testid="message-textarea"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
      />
      <button data-testid="send-button" onClick={onSend} disabled={disabled}>
        Send
      </button>
      {hasAttachment && <div data-testid="has-attachment">Attachment</div>}
    </div>
  ),
}));

beforeEach(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }),
  });
  global.URL.createObjectURL = vi.fn().mockReturnValue("blob:test-url");
  global.URL.revokeObjectURL = vi.fn();
  capturedOnImageUpload = null;
  vi.clearAllMocks();
});

const renderCompare = (options: { searchParams?: string; onUrlUpdate?: OnUrlUpdateFunction } = {}) =>
  render(<CompareUI accessToken="test-token" disabledPersonalKeyCreation={false} />, options);

const panelModels = () =>
  screen.queryAllByTestId(/^comparison-panel-/).map((panel) => panel.getAttribute("data-model"));

const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) => {
  const update = onUrlUpdate.mock.calls.at(-1)?.[0];
  if (!update) throw new Error("expected a URL update");
  return update;
};

describe("CompareUI", () => {
  it("should render", () => {
    render(<CompareUI accessToken="test-token" disabledPersonalKeyCreation={false} />);
    expect(screen.getByTestId("comparison-panel-1")).toBeInTheDocument();
    expect(screen.getByTestId("comparison-panel-2")).toBeInTheDocument();
    expect(screen.getByTestId("message-input")).toBeInTheDocument();
  });

  it("adds a comparison when Add Comparison button is clicked", async () => {
    const user = userEvent.setup();
    const { container } = render(<CompareUI accessToken="test-token" disabledPersonalKeyCreation={false} />);

    // Verify initial state: 2 comparison panels
    expect(screen.getByTestId("comparison-panel-1")).toBeInTheDocument();
    expect(screen.getByTestId("comparison-panel-2")).toBeInTheDocument();
    let comparisonPanels = container.querySelectorAll('[data-testid^="comparison-panel-"]');
    expect(comparisonPanels).toHaveLength(2);

    await user.click(screen.getByRole("button", { name: /Add Comparison/i }));

    // Wait for the new comparison panel to be added (should have 3 total now)
    await waitFor(() => {
      comparisonPanels = container.querySelectorAll('[data-testid^="comparison-panel-"]');
      expect(comparisonPanels).toHaveLength(3);
    });

    // Verify the original 2 panels are still there
    expect(screen.getByTestId("comparison-panel-1")).toBeInTheDocument();
    expect(screen.getByTestId("comparison-panel-2")).toBeInTheDocument();
  });

  it("should handle image upload and send message with attachment", async () => {
    const user = userEvent.setup();
    render(<CompareUI accessToken="test-token" disabledPersonalKeyCreation={false} />);

    const file = new File(["test content"], "test-image.png", { type: "image/png" });

    await waitFor(() => {
      expect(capturedOnImageUpload).not.toBeNull();
    });

    if (capturedOnImageUpload) {
      capturedOnImageUpload(file);
    }

    await waitFor(() => {
      expect(screen.getByTestId("has-attachment")).toBeInTheDocument();
    });

    const textarea = screen.getByTestId("message-textarea");
    fireEvent.change(textarea, { target: { value: "Describe this image" } });

    const sendButton = screen.getByTestId("send-button");
    expect(sendButton).toBeEnabled();
    await user.click(sendButton);

    await waitFor(() => {
      expect(makeOpenAIChatCompletionRequest).toHaveBeenCalled();
    });
  });

  describe("URL state", () => {
    it("fills two panels with the loaded models when the URL is empty", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderCompare({ onUrlUpdate });

      await waitFor(() => expect(panelModels()).toEqual(["gpt-4", "gpt-3.5-turbo"]));
      expect(onUrlUpdate).not.toHaveBeenCalled();
    });

    it("opens one panel per model listed in the URL, in order", async () => {
      renderCompare({ searchParams: "?cmp_models=gpt-3.5-turbo,gpt-4,gpt-3.5-turbo" });

      await waitFor(() => expect(panelModels()).toEqual(["gpt-3.5-turbo", "gpt-4", "gpt-3.5-turbo"]));
      expect(screen.getByRole("button", { name: /Add Comparison/i })).toBeDisabled();
    });

    it("opens a single panel for a single URL model", async () => {
      renderCompare({ searchParams: "?cmp_models=gpt-3.5-turbo" });

      await waitFor(() => expect(panelModels()).toEqual(["gpt-3.5-turbo"]));
    });

    it("swaps a URL model the key cannot use for an available one", async () => {
      renderCompare({ searchParams: "?cmp_models=gpt-3.5-turbo,retired-model" });

      await waitFor(() => expect(panelModels()).toEqual(["gpt-3.5-turbo", "gpt-3.5-turbo"]));
    });

    it("writes a picked model into its panel's slot", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderCompare({ searchParams: "?cmp_models=gpt-3.5-turbo,gpt-4,gpt-4", onUrlUpdate });
      await waitFor(() => expect(panelModels()).toEqual(["gpt-3.5-turbo", "gpt-4", "gpt-4"]));

      await user.click(screen.getAllByTestId(/^pick-/)[1]);

      await waitFor(() =>
        expect(lastUrlUpdate(onUrlUpdate).searchParams.get("cmp_models")).toBe("gpt-3.5-turbo,gpt-3.5-turbo,gpt-4"),
      );
      expect(panelModels()).toEqual(["gpt-3.5-turbo", "gpt-3.5-turbo", "gpt-4"]);
    });

    it("appends the new panel's model when a comparison is added", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderCompare({ onUrlUpdate });
      await waitFor(() => expect(panelModels()).toEqual(["gpt-4", "gpt-3.5-turbo"]));

      await user.click(screen.getByRole("button", { name: /Add Comparison/i }));

      await waitFor(() =>
        expect(lastUrlUpdate(onUrlUpdate).searchParams.get("cmp_models")).toBe("gpt-4,gpt-3.5-turbo,gpt-4"),
      );
      expect(panelModels()).toEqual(["gpt-4", "gpt-3.5-turbo", "gpt-4"]);
    });

    it("drops the removed panel's slot and keeps the others", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderCompare({ searchParams: "?cmp_models=gpt-4,gpt-3.5-turbo,gpt-4", onUrlUpdate });
      await waitFor(() => expect(panelModels()).toEqual(["gpt-4", "gpt-3.5-turbo", "gpt-4"]));

      await user.click(screen.getByTestId("remove-1"));

      await waitFor(() =>
        expect(lastUrlUpdate(onUrlUpdate).searchParams.get("cmp_models")).toBe("gpt-3.5-turbo,gpt-4"),
      );
      expect(panelModels()).toEqual(["gpt-3.5-turbo", "gpt-4"]);
      expect(screen.getByTestId("comparison-panel-2")).toBeInTheDocument();
      expect(screen.getByTestId("comparison-panel-3")).toBeInTheDocument();
    });

    it("opens the endpoint named in the URL", async () => {
      renderCompare({ searchParams: "?cmp_endpoint=/a2a" });

      expect(await screen.findByRole("combobox", { name: "Endpoint" })).toHaveTextContent("/a2a (Agents)");
    });

    it("ignores an unknown URL endpoint", async () => {
      renderCompare({ searchParams: "?cmp_endpoint=/v1/unknown" });

      expect(await screen.findByRole("combobox", { name: "Endpoint" })).toHaveTextContent("/v1/chat/completions");
    });

    it("writes the chosen endpoint to the URL", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderCompare({ onUrlUpdate });

      await chooseSelectOption(user, screen.getByRole("combobox", { name: "Endpoint" }), "/a2a (Agents)");

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate).searchParams.get("cmp_endpoint")).toBe("/a2a"));
      expect(screen.getByRole("combobox", { name: "Endpoint" })).toHaveTextContent("/a2a (Agents)");
    });
  });
});
