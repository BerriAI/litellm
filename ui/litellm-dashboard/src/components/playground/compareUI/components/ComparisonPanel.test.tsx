import { render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ComparisonInstance } from "../CompareUI";
import { ComparisonPanel } from "./ComparisonPanel";
import { EndpointId, ENDPOINT_CONFIGS } from "../endpoint_config";

vi.mock("./MessageDisplay", () => ({
  MessageDisplay: () => <div data-testid="message-display">MessageDisplay</div>,
}));

vi.mock("./UnifiedSelector", () => ({
  UnifiedSelector: ({ value, onChange }: { value: string; onChange: (val: string) => void }) => (
    <select data-testid="unified-selector" value={value} onChange={(e) => onChange(e.target.value)}>
      <option value="">Select option</option>
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
  agent: "",
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
  selectorOptions: [
    { value: "gpt-4", label: "gpt-4" },
    { value: "gpt-3.5-turbo", label: "gpt-3.5-turbo" },
  ],
  isLoadingOptions: false,
  endpointConfig: ENDPOINT_CONFIGS[EndpointId.CHAT_COMPLETIONS],
  apiKey: "test-api-key",
};

describe("ComparisonPanel", () => {
  it("should render", () => {
    const { getByTestId } = render(<ComparisonPanel {...mockProps} />);
    expect(getByTestId("unified-selector")).toBeInTheDocument();
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
