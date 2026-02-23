import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { GuardrailTestPanel } from "./GuardrailTestPanel";

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

describe("GuardrailTestPanel", () => {
  const mockOnSubmit = vi.fn();
  const mockOnClose = vi.fn();
  const mockGuardrailNames = ["test-guardrail-1", "test-guardrail-2"];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should submit text when Enter key is pressed", async () => {
    /**
     * Tests that pressing Enter submits the form with the input text.
     * This validates the keyboard shortcut functionality.
     */
    const user = userEvent.setup();

    render(
      <GuardrailTestPanel
        guardrailNames={mockGuardrailNames}
        onSubmit={mockOnSubmit}
        isLoading={false}
        results={null}
        errors={null}
        onClose={mockOnClose}
      />
    );

    // Find and type in the textarea
    const textarea = screen.getByPlaceholderText("Enter text to test with guardrails...");
    await user.type(textarea, "Test input text");

    // Press Enter to submit
    await user.keyboard("{Enter}");

    // Verify onSubmit was called with the correct text
    await waitFor(() => {
      expect(mockOnSubmit).toHaveBeenCalledWith("Test input text");
    });
  });
});

