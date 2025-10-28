import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import GuardrailSelector from "./GuardrailSelector";
import * as networking from "../networking";

vi.mock("../networking");

describe("GuardrailSelector", () => {
  const mockAccessToken = "test-token";
  const mockOnChange = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should load and display guardrails from API", async () => {
    /**
     * Tests that the selector fetches guardrails from the API and displays them.
     * This validates the core data loading functionality.
     */
    const mockGuardrails = [
      {
        guardrail_name: "pii-guard",
        litellm_params: { guardrail: "presidio", mode: "pre_call", default_on: false },
      },
      {
        guardrail_name: "content-filter",
        litellm_params: { guardrail: "lakera", mode: "post_call", default_on: true },
      },
    ];

    vi.mocked(networking.getGuardrailsList).mockResolvedValue({
      guardrails: mockGuardrails,
    });

    render(
      <GuardrailSelector
        accessToken={mockAccessToken}
        onChange={mockOnChange}
        value={[]}
      />
    );

    // Wait for guardrails to load
    await waitFor(() => {
      expect(networking.getGuardrailsList).toHaveBeenCalledWith(mockAccessToken);
    });

    // Click to open the dropdown
    const select = screen.getByRole("combobox");
    await userEvent.click(select);

    // Verify guardrails are displayed
    await waitFor(() => {
      expect(screen.getByText("pii-guard")).toBeInTheDocument();
      expect(screen.getByText("content-filter")).toBeInTheDocument();
    });
  });
});

