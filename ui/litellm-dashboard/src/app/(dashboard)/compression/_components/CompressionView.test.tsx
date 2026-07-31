import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { mockCreateGuardrailCall, mockGetGuardrailsList } = vi.hoisted(() => ({
  mockCreateGuardrailCall: vi.fn(),
  mockGetGuardrailsList: vi.fn(),
}));

vi.mock("@/components/networking", () => ({
  createGuardrailCall: mockCreateGuardrailCall,
  getGuardrailsList: mockGetGuardrailsList,
}));

vi.mock("@/components/molecules/notifications_manager", () => ({
  default: { success: vi.fn(), fromBackend: vi.fn() },
}));

import CompressionView from "./CompressionView";

const headroomGuardrail = {
  guardrail_id: "g-1",
  guardrail_name: "headroom-prod",
  litellm_params: { guardrail: "headroom", api_base: "https://headroom.internal", default_on: true },
};

const piiGuardrail = {
  guardrail_id: "g-2",
  guardrail_name: "presidio-pii",
  litellm_params: { guardrail: "presidio", api_base: "https://presidio.internal", default_on: false },
};

describe("CompressionView", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetGuardrailsList.mockResolvedValue({ guardrails: [headroomGuardrail, piiGuardrail] });
    mockCreateGuardrailCall.mockResolvedValue({});
  });

  it("lists only compression guardrails with their always-on state", async () => {
    render(<CompressionView accessToken="test-token" />);

    await waitFor(() => expect(screen.getByText("headroom-prod")).toBeInTheDocument());
    expect(screen.getByText("https://headroom.internal")).toBeInTheDocument();
    expect(screen.getByText("Always on")).toBeInTheDocument();
    expect(screen.queryByText("presidio-pii")).not.toBeInTheDocument();
  });

  it("creates a headroom guardrail from the form and reloads the list", async () => {
    render(<CompressionView accessToken="test-token" />);
    await waitFor(() => expect(mockGetGuardrailsList).toHaveBeenCalledTimes(1));

    fireEvent.change(screen.getByPlaceholderText("headroom-compression"), { target: { value: "headroom-new" } });
    fireEvent.change(screen.getByPlaceholderText("https://your-headroom-endpoint"), {
      target: { value: "https://new-headroom.internal" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Add guardrail" }));

    await waitFor(() =>
      expect(mockCreateGuardrailCall).toHaveBeenCalledWith("test-token", {
        guardrail_name: "headroom-new",
        litellm_params: {
          guardrail: "headroom",
          mode: "pre_call",
          api_base: "https://new-headroom.internal",
          default_on: true,
        },
      }),
    );
    await waitFor(() => expect(mockGetGuardrailsList).toHaveBeenCalledTimes(2));
  });

  it("sends default_on false when 'Apply to all requests' is switched off", async () => {
    render(<CompressionView accessToken="test-token" />);
    await waitFor(() => expect(mockGetGuardrailsList).toHaveBeenCalledTimes(1));

    fireEvent.change(screen.getByPlaceholderText("headroom-compression"), { target: { value: "headroom-optin" } });
    fireEvent.change(screen.getByPlaceholderText("https://your-headroom-endpoint"), {
      target: { value: "https://optin-headroom.internal" },
    });
    fireEvent.click(screen.getByRole("switch"));
    fireEvent.click(screen.getByRole("button", { name: "Add guardrail" }));

    await waitFor(() =>
      expect(mockCreateGuardrailCall).toHaveBeenCalledWith(
        "test-token",
        expect.objectContaining({
          litellm_params: expect.objectContaining({ default_on: false }),
        }),
      ),
    );
  });

  it("does not call the backend without an access token", async () => {
    render(<CompressionView accessToken={null} />);

    await waitFor(() => expect(screen.getByRole("button", { name: "Add guardrail" })).toBeInTheDocument());
    expect(mockGetGuardrailsList).not.toHaveBeenCalled();
  });
});
