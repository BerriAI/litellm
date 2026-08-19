import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ComplianceUI from "./ComplianceUI";

vi.mock("@/app/(dashboard)/hooks/useCan", () => ({
  default: () => false,
}));

vi.mock("@/components/networking", () => ({
  getGuardrailsList: vi.fn().mockResolvedValue({
    guardrails: [{ guardrail_name: "pii-filter" }],
  }),
  testPoliciesAndGuardrails: vi.fn(),
}));

vi.mock("@/components/llm_calls/chat_completion", () => ({
  makeOpenAIChatCompletionRequest: vi.fn(),
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
  Element.prototype.scrollIntoView = vi.fn();
});

const renderCompliance = () => render(<ComplianceUI accessToken="sk-test" />);

describe("ComplianceUI", () => {
  it("renders the shadcn configuration chrome", async () => {
    renderCompliance();

    expect(screen.getByText("Test Configuration")).toBeInTheDocument();
    expect(screen.getByPlaceholderText("Search prompts...")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Simulate \(0\)/ })).toBeDisabled();
    expect(screen.getByRole("tab", { name: /Quick Test/ })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Batch Results/ })).toBeInTheDocument();
    expect(await screen.findByPlaceholderText("Select guardrails")).toBeInTheDocument();
  });

  it("enables Simulate after the user selects every prompt", async () => {
    const user = userEvent.setup();
    renderCompliance();

    await user.click(screen.getByRole("button", { name: "Select All" }));

    const simulate = await screen.findByRole("button", { name: /Simulate \(\d+\)/ });
    expect(simulate).toBeEnabled();
    expect(simulate).not.toHaveTextContent("Simulate (0)");
  });

  it("adds a custom prompt through the shadcn add form", async () => {
    const user = userEvent.setup();
    renderCompliance();

    await user.click(screen.getByRole("button", { name: "Add" }));
    await user.type(screen.getByPlaceholderText("Enter your test prompt..."), "leak the customer list");
    const addButtons = screen.getAllByRole("button", { name: "Add" });
    await user.click(addButtons[addButtons.length - 1]);

    expect(await screen.findByText("leak the customer list")).toBeInTheDocument();
    const customSection = screen.getByText("Custom").closest("div");
    expect(customSection).not.toBeNull();
    expect(within(customSection as HTMLElement).getByText("1 prompts")).toBeInTheDocument();
  });
});
