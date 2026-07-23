import { render, screen, fireEvent, within } from "@testing-library/react";
import { beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { TeamGuardrailsTab } from "./TeamGuardrailsTab";
import { listGuardrailSubmissions, type GuardrailSubmissionItem } from "@/components/networking";

vi.mock("@/components/networking", () => ({
  listGuardrailSubmissions: vi.fn(),
  approveGuardrailSubmission: vi.fn(),
  rejectGuardrailSubmission: vi.fn(),
  updateGuardrailCall: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/guardrails/useRegisterGuardrail", () => ({
  useRegisterGuardrail: () => ({ mutateAsync: vi.fn(), isPending: false }),
}));

const pendingSubmission: GuardrailSubmissionItem = {
  guardrail_id: "gd-1",
  guardrail_name: "PII Scanner",
  status: "pending_review",
  team_id: "team-alpha",
  team_guardrail: true,
  litellm_params: {
    api_base: "https://guard.example.com/scan",
    forward_api_key: true,
    headers: [{ key: "X-Api-Key", value: "secret" }],
    extra_headers: ["x-request-id"],
    method: "POST",
  },
  guardrail_info: { description: "Scans for PII", model: "gpt-4o" },
  submitted_by_email: "member@example.com",
  submitted_at: "2026-05-01T00:00:00Z",
};

beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
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
});

beforeEach(() => {
  vi.mocked(listGuardrailSubmissions).mockResolvedValue({
    submissions: [pendingSubmission],
    summary: { total: 1, pending_review: 1, active: 0, rejected: 0 },
  });
});

async function renderTab(userRole?: string) {
  render(<TeamGuardrailsTab accessToken="sk-test" userRole={userRole} />);
  return screen.findByText("PII Scanner");
}

describe("TeamGuardrailsTab role gating", () => {
  it("shows submission status and team to a non-admin submitter", async () => {
    const heading = await renderTab("Internal User");
    const card = heading.closest(".bg-white") as HTMLElement;
    // Point 1: the submitter can see the status after submitting
    expect(within(card).getByText("Pending Review")).toBeInTheDocument();
    // Point 2: the team member can see which team the guardrail is attached to
    expect(within(card).getByText("Team: team-alpha")).toBeInTheDocument();
  });

  it("hides Approve/Reject and the forward-key toggle from a non-admin", async () => {
    await renderTab("Internal User");
    expect(screen.queryByRole("button", { name: "Approve" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Reject" })).not.toBeInTheDocument();
    expect(screen.queryByText("Forward API Key")).not.toBeInTheDocument();
  });

  it("hides Approve/Reject from an Admin Viewer (backend rejects them too)", async () => {
    await renderTab("Admin Viewer");
    expect(screen.queryByRole("button", { name: "Approve" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Reject" })).not.toBeInTheDocument();
  });

  it("shows Approve/Reject and the forward-key toggle to a proxy admin", async () => {
    await renderTab("Admin");
    expect(screen.getByRole("button", { name: "Approve" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reject" })).toBeInTheDocument();
    expect(screen.getByText("Forward API Key")).toBeInTheDocument();
  });

  it("shows header editors in the detail panel only to a proxy admin", async () => {
    await renderTab("Admin");
    fireEvent.click(screen.getByRole("button", { name: "Review" }));
    const panel = screen.getByText("Forward LiteLLM API Key").closest("div");
    expect(panel).not.toBeNull();
    expect(screen.getByPlaceholderText("Header name (e.g. X-API-Key)")).toBeInTheDocument();
    // Forward-key is an interactive toggle for admins, not a static label
    expect(screen.queryByText(/^(Enabled|Disabled)$/)).not.toBeInTheDocument();
  });

  it("renders the detail panel read-only for a non-admin", async () => {
    await renderTab("Internal User");
    fireEvent.click(screen.getByRole("button", { name: "Review" }));
    // Configured headers are still visible to read
    expect(screen.getByText("X-Api-Key: secret")).toBeInTheDocument();
    expect(screen.getByText("x-request-id")).toBeInTheDocument();
    // But no editing affordances
    expect(screen.queryByPlaceholderText("Header name (e.g. X-API-Key)")).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText("e.g. x-request-id")).not.toBeInTheDocument();
    // Forward-key renders as a read-only state, not a toggle
    expect(screen.getByText("Enabled")).toBeInTheDocument();
  });
});
