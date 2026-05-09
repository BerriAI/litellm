import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderWithProviders, screen } from "./test-utils";
import { TeamGuardrailsTab } from "../src/components/guardrails/TeamGuardrailsTab";

vi.mock("../src/components/networking", () => ({
  listGuardrailSubmissions: vi.fn(),
  approveGuardrailSubmission: vi.fn(),
  rejectGuardrailSubmission: vi.fn(),
  updateGuardrailCall: vi.fn(),
}));

vi.mock("../src/app/(dashboard)/hooks/guardrails/useRegisterGuardrail", () => ({
  useRegisterGuardrail: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
}));

vi.mock("../src/components/common_components/team_dropdown", () => ({
  default: () => null,
}));

import { listGuardrailSubmissions } from "../src/components/networking";

const pendingSubmission = {
  guardrail_id: "guard-1",
  guardrail_name: "test-pending-guardrail",
  status: "pending_review",
  team_id: "team-1",
  team_guardrail: true,
  litellm_params: {
    guardrail: "generic_guardrail_api",
    mode: "pre_call",
    api_base: "https://example.com/guard",
  },
  guardrail_info: {},
  submitted_at: "2026-05-09T00:00:00Z",
};

describe("TeamGuardrailsTab — approve/reject role gate", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(listGuardrailSubmissions).mockResolvedValue({
      submissions: [pendingSubmission],
      summary: { total: 1, pending_review: 1, active: 0, rejected: 0 },
    });
  });

  it("hides Approve and Reject buttons for an internal user on a pending submission", async () => {
    renderWithProviders(
      <TeamGuardrailsTab accessToken="test-token" userRole="Internal User" />
    );

    // Wait for the submission row to render
    await screen.findByText("test-pending-guardrail");

    expect(screen.queryByRole("button", { name: /approve/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /reject/i })).not.toBeInTheDocument();
  });

  it("shows Approve and Reject buttons for an admin on a pending submission", async () => {
    renderWithProviders(
      <TeamGuardrailsTab accessToken="test-token" userRole="Admin" />
    );

    await screen.findByText("test-pending-guardrail");

    expect(screen.getByRole("button", { name: /approve/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /reject/i })).toBeInTheDocument();
  });

  it("hides Approve and Reject buttons when userRole is undefined (defaults to non-admin)", async () => {
    renderWithProviders(<TeamGuardrailsTab accessToken="test-token" />);

    await screen.findByText("test-pending-guardrail");

    expect(screen.queryByRole("button", { name: /approve/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /reject/i })).not.toBeInTheDocument();
  });
});
