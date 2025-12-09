import { render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ComparisonInstance } from "../CompareUI";
import { ComparisonPanel } from "./ComparisonPanel";

vi.mock("./MessageDisplay", () => ({
  MessageDisplay: () => <div data-testid="message-display">MessageDisplay</div>,
}));

vi.mock("./ModelSelector", () => ({
  ModelSelector: ({ value, onChange }: { value: string; onChange: (val: string) => void }) => (
    <select data-testid="model-selector" value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">Select model</option>
      <option value="gpt-4">gpt-4</option>
    </select>
  ),
}));

vi.mock("../../../tag_management/TagSelector", () => ({
  default: () => <div data-testid="tag-selector">TagSelector</div>,
}));

vi.mock("../../../vector_store_management/VectorStoreSelector", () => ({
  default: () => <div data-testid="vector-store-selector">VectorStoreSelector</div>,
}));

vi.mock("../../../guardrails/GuardrailSelector", () => ({
  default: () => <div data-testid="guardrail-selector">GuardrailSelector</div>,
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

const mockComparison: ComparisonInstance = {
  id: "1",
  model: "gpt-4",
  messages: [],
  isLoading: false,
  tags: [],
  mcpTools: [],
  vectorStores: [],
  guardrails: [],
  temperature: 1,
  maxTokens: 2048,
  applyAcrossModels: false,
  useAdvancedParams: false,
};

const mockProps = {
  comparison: mockComparison,
  onUpdate: vi.fn(),
  onRemove: vi.fn(),
  canRemove: true,
  modelOptions: ["gpt-4", "gpt-3.5-turbo"],
  isLoadingModels: false,
  apiKey: "test-api-key",
};

describe("ComparisonPanel", () => {
  it("should render", () => {
    const { getByTestId } = render(<ComparisonPanel {...mockProps} />);
    expect(getByTestId("model-selector")).toBeInTheDocument();
    expect(getByTestId("message-display")).toBeInTheDocument();
  });

  it("should call onRemove when remove button is clicked", async () => {
    const user = userEvent.setup();
    const onRemove = vi.fn();
    const { container } = render(<ComparisonPanel {...mockProps} onRemove={onRemove} />);
    const removeButton = container.querySelector('button[class*="text-red-600"]');
    expect(removeButton).toBeInTheDocument();
    await user.click(removeButton!);
    expect(onRemove).toHaveBeenCalledTimes(1);
  });
});
