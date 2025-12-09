import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { GuardrailTestResults } from "./GuardrailTestResults";

Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: vi.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

describe("GuardrailTestResults", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should collapse and expand results when clicked", async () => {
    /**
     * Tests that clicking on a result header toggles its collapsed state.
     * This validates the core collapse/expand functionality for managing large payloads.
     */
    const user = userEvent.setup();
    const mockResults = [
      {
        guardrailName: "test-guardrail",
        response_text: "This is a very long response text that should be collapsible",
        latency: 250,
      },
    ];

    render(<GuardrailTestResults results={mockResults} errors={null} />);

    // Verify output text is initially visible
    expect(screen.getByText("This is a very long response text that should be collapsible")).toBeInTheDocument();

    // Click on the guardrail name to collapse
    const guardrailHeader = screen.getByText("test-guardrail");
    await user.click(guardrailHeader);

    // Verify output text is now hidden
    expect(screen.queryByText("This is a very long response text that should be collapsible")).not.toBeInTheDocument();

    // Click again to expand
    await user.click(guardrailHeader);

    // Verify output text is visible again
    expect(screen.getByText("This is a very long response text that should be collapsible")).toBeInTheDocument();
  });
});

