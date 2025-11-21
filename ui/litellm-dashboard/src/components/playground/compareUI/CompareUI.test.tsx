import { render, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import CompareUI from "./CompareUI";

vi.mock("../llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([{ model_group: "gpt-4" }, { model_group: "gpt-3.5-turbo" }]),
}));

vi.mock("../llm_calls/chat_completion", () => ({
  makeOpenAIChatCompletionRequest: vi.fn().mockResolvedValue(undefined),
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
  MessageInput: ({ value, onChange, onSend, disabled }: any) => (
    <div data-testid="message-input">
      <textarea
        data-testid="message-textarea"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
      />
      <button data-testid="send-button" onClick={onSend} disabled={disabled}>
        Send
      </button>
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
});
