import { render, screen, waitFor } from "@testing-library/react";
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

vi.mock("@/components/tag_management/TagSelector", () => ({
  default: () => <div data-testid="tag-selector">TagSelector</div>,
}));

vi.mock("@/components/vector_store_management/VectorStoreSelector", () => ({
  default: () => <div data-testid="vector-store-selector">VectorStoreSelector</div>,
}));

vi.mock("@/components/guardrails/GuardrailSelector", () => ({
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

const buttonWithIcon = (icon: string): HTMLButtonElement => {
  const match = Array.from(document.querySelectorAll("button")).find((button) =>
    button.querySelector(`svg.lucide-${icon}`),
  );
  if (!match) throw new Error(`no button carrying the ${icon} icon`);
  return match;
};

const openSettings = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(buttonWithIcon("settings"));
  await screen.findByText("General Settings");
};

describe("ComparisonPanel", () => {
  it("renders the selector and the transcript", () => {
    render(<ComparisonPanel {...mockProps} />);

    expect(screen.getByTestId("unified-selector")).toBeInTheDocument();
    expect(screen.getByTestId("message-display")).toBeInTheDocument();
  });

  it("removes the panel when the remove control is used", async () => {
    const user = userEvent.setup();
    const onRemove = vi.fn();
    render(<ComparisonPanel {...mockProps} onRemove={onRemove} />);

    await user.click(buttonWithIcon("x"));

    expect(onRemove).toHaveBeenCalledTimes(1);
  });

  it("hides the remove control on the last remaining panel", () => {
    render(<ComparisonPanel {...mockProps} canRemove={false} />);

    expect(() => buttonWithIcon("x")).toThrow();
  });

  it("keeps the settings out of sight until the gear is used", async () => {
    const user = userEvent.setup();
    render(<ComparisonPanel {...mockProps} />);

    expect(screen.queryByText("General Settings")).not.toBeInTheDocument();

    await openSettings(user);

    expect(screen.getByText("General Settings")).toBeInTheDocument();
    expect(screen.getByText("Advanced Settings")).toBeInTheDocument();
    expect(screen.getByTestId("tag-selector")).toBeInTheDocument();
    expect(screen.getByTestId("vector-store-selector")).toBeInTheDocument();
    expect(screen.getByTestId("guardrail-selector")).toBeInTheDocument();
  });

  it("shows the current temperature and token ceiling", async () => {
    const user = userEvent.setup();
    render(<ComparisonPanel {...mockProps} />);

    await openSettings(user);

    expect(screen.getByText("Temperature")).toBeInTheDocument();
    expect(screen.getByText("1.00")).toBeInTheDocument();
    expect(screen.getByText("Max Tokens")).toBeInTheDocument();
    expect(screen.getByText("2048")).toBeInTheDocument();

    const ranges = Array.from(document.querySelectorAll("[aria-valuenow]"));
    expect(ranges.map((range) => range.getAttribute("aria-valuenow"))).toEqual(["1", "2048"]);
  });

  it("pushes the whole parameter set to every panel when sync is switched on", async () => {
    const user = userEvent.setup();
    const onUpdate = vi.fn();
    render(<ComparisonPanel {...mockProps} onUpdate={onUpdate} />);

    await openSettings(user);
    await user.click(screen.getByRole("checkbox", { name: /Sync Settings Across Models/i }));

    await waitFor(() => expect(onUpdate).toHaveBeenCalled());
    const [updates, options] = onUpdate.mock.calls[0];
    expect(updates.applyAcrossModels).toBe(true);
    expect(updates.temperature).toBe(1);
    expect(updates.maxTokens).toBe(2048);
    expect(options.applyToAll).toBe(true);
    expect(options.keysToApply).toContain("temperature");
    expect(options.keysToApply).toContain("maxTokens");
  });

  it("turns sync off without resetting the values it was sharing", async () => {
    const user = userEvent.setup();
    const onUpdate = vi.fn();
    render(
      <ComparisonPanel
        {...mockProps}
        comparison={{ ...mockComparison, applyAcrossModels: true }}
        onUpdate={onUpdate}
      />,
    );

    await openSettings(user);
    await user.click(screen.getByRole("checkbox", { name: /Sync Settings Across Models/i }));

    await waitFor(() => expect(onUpdate).toHaveBeenCalled());
    expect(onUpdate.mock.calls[0][0]).toEqual({ applyAcrossModels: false });
  });

  it("keeps an advanced-parameter toggle local while sync is off", async () => {
    const user = userEvent.setup();
    const onUpdate = vi.fn();
    render(<ComparisonPanel {...mockProps} onUpdate={onUpdate} />);

    await openSettings(user);
    await user.click(screen.getByRole("checkbox", { name: /Use Advanced Parameters/i }));

    await waitFor(() => expect(onUpdate).toHaveBeenCalled());
    expect(onUpdate.mock.calls[0][0]).toEqual({ useAdvancedParams: true });
    expect(onUpdate.mock.calls[0][1]).toBeUndefined();
  });

  it("fans an advanced-parameter toggle out to every panel while sync is on", async () => {
    const user = userEvent.setup();
    const onUpdate = vi.fn();
    render(
      <ComparisonPanel
        {...mockProps}
        comparison={{ ...mockComparison, applyAcrossModels: true }}
        onUpdate={onUpdate}
      />,
    );

    await openSettings(user);
    await user.click(screen.getByRole("checkbox", { name: /Use Advanced Parameters/i }));

    await waitFor(() => expect(onUpdate).toHaveBeenCalled());
    expect(onUpdate.mock.calls[0][1]).toEqual({ applyToAll: true, keysToApply: ["useAdvancedParams"] });
  });
});
