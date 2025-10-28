import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import GuardrailSelector from "./GuardrailSelector";
import * as networking from "../networking";

vi.mock("../networking");

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

describe("GuardrailSelector", () => {
  const mockAccessToken = "test-token";
  const mockOnChange = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should load guardrails from API when component mounts", async () => {
    /**
     * Tests that the selector fetches guardrails from the API on mount.
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

    // Verify API was called with correct token
    await waitFor(() => {
      expect(networking.getGuardrailsList).toHaveBeenCalledWith(mockAccessToken);
    });
  });
});

