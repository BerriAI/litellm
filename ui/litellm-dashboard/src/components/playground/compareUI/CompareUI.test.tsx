import { render, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import CompareUI from "./CompareUI";
import { makeOpenAIChatCompletionRequest } from "../llm_calls/chat_completion";

vi.mock("../llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([{ model_group: "gpt-4" }, { model_group: "gpt-3.5-turbo" }]),
}));

vi.mock("../llm_calls/chat_completion", () => ({
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
  ComparisonPanel: ({ comparison, onRemove }: { comparison: any; onRemove: () => void }) => (
    <div data-testid={`comparison-panel-${comparison.id}`}>
      <button data-testid={`remove-${comparison.id}`} onClick={onRemove}>
        Remove
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

describe("CompareUI", () => {
  it("should render", () => {
    const { getByTestId } = render(<CompareUI accessToken="test-token" disabledPersonalKeyCreation={false} />);
    expect(getByTestId("comparison-panel-1")).toBeInTheDocument();
    expect(getByTestId("comparison-panel-2")).toBeInTheDocument();
    expect(getByTestId("message-input")).toBeInTheDocument();
  });

  it("adds a comparison when Add Comparison button is clicked", async () => {
    const user = userEvent.setup();
    const { container, getByTestId } = render(
      <CompareUI accessToken="test-token" disabledPersonalKeyCreation={false} />,
    );

    // Verify initial state: 2 comparison panels
    expect(getByTestId("comparison-panel-1")).toBeInTheDocument();
    expect(getByTestId("comparison-panel-2")).toBeInTheDocument();
    let comparisonPanels = container.querySelectorAll('[data-testid^="comparison-panel-"]');
    expect(comparisonPanels).toHaveLength(2);

    const addButtons = Array.from(container.querySelectorAll('button[class*="ant-btn"]'));
    const addComparisonButton = addButtons.find((btn) => btn.textContent?.includes("Add Comparison"));
    expect(addComparisonButton).toBeInTheDocument();
    await user.click(addComparisonButton!);

    // Wait for the new comparison panel to be added (should have 3 total now)
    await waitFor(() => {
      comparisonPanels = container.querySelectorAll('[data-testid^="comparison-panel-"]');
      expect(comparisonPanels).toHaveLength(3);
    });

    // Verify the original 2 panels are still there
    expect(getByTestId("comparison-panel-1")).toBeInTheDocument();
    expect(getByTestId("comparison-panel-2")).toBeInTheDocument();
  });

  it("should handle image upload and send message with attachment", async () => {
    const user = userEvent.setup();
    const { getByTestId, queryByTestId } = render(
      <CompareUI accessToken="test-token" disabledPersonalKeyCreation={false} />,
    );

    const file = new File(["test content"], "test-image.png", { type: "image/png" });
    
    await waitFor(() => {
      expect(capturedOnImageUpload).not.toBeNull();
    });

    if (capturedOnImageUpload) {
      capturedOnImageUpload(file);
    }

    await waitFor(() => {
      expect(queryByTestId("has-attachment")).toBeInTheDocument();
    });

    const textarea = getByTestId("message-textarea");
    await user.type(textarea, "Describe this image");

    const sendButton = getByTestId("send-button");
    expect(sendButton).not.toBeDisabled();
    await user.click(sendButton);

    await waitFor(() => {
      expect(makeOpenAIChatCompletionRequest).toHaveBeenCalled();
    });
  });
});
