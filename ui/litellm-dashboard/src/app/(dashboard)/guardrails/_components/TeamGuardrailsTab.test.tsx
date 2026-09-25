import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import { screen, fireEvent, waitFor } from "@testing-library/react";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { TeamGuardrailsTab } from "./TeamGuardrailsTab";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

vi.mock("@/components/networking", () => ({
  listGuardrailSubmissions: vi.fn(),
  approveGuardrailSubmission: vi.fn(),
  rejectGuardrailSubmission: vi.fn(),
  updateGuardrailCall: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/guardrails/useRegisterGuardrail", () => ({
  useRegisterGuardrail: () => ({
    mutateAsync: vi.fn(),
    isPending: false,
  }),
}));

vi.mock("@/components/common_components/team_dropdown", () => ({
  default: () => null,
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: vi.fn(),
}));

import {
  approveGuardrailSubmission,
  listGuardrailSubmissions,
  rejectGuardrailSubmission,
} from "@/components/networking";

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
    headers: { "X-API-Key": "secret" },
    extra_headers: ["x-request-id"],
  },
  guardrail_info: {},
  submitted_at: "2026-05-09T00:00:00Z",
};

const baseAuth: Omit<ReturnType<typeof useAuthorized>, "userRole"> = {
  isLoading: false,
  isAuthorized: true,
  token: "test-token",
  accessToken: "test-token",
  userId: "user-1",
  userEmail: "user@example.com",
  userRoleLabel: "",
  isViewOnly: false,
  premiumUser: false,
  disabledPersonalKeyCreation: null,
  showSSOBanner: false,
};

describe("TeamGuardrailsTab — approve/reject role gate", () => {
  const mockUseAuthorized = vi.mocked(useAuthorized);

  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(listGuardrailSubmissions).mockResolvedValue({
      submissions: [pendingSubmission],
      summary: { total: 1, pending_review: 1, active: 0, rejected: 0 },
    });
  });

  it("hides Approve and Reject buttons for an internal user on a pending submission", async () => {
    mockUseAuthorized.mockReturnValue({ ...baseAuth, userRole: "Internal User" });
    renderWithProviders(<TeamGuardrailsTab accessToken="test-token" />);

    await screen.findByText("test-pending-guardrail");

    expect(screen.queryByRole("button", { name: /approve/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /reject/i })).not.toBeInTheDocument();
  });

  it("hides Approve and Reject buttons for an Admin Viewer, whom the backend rejects with 403", async () => {
    mockUseAuthorized.mockReturnValue({ ...baseAuth, userRole: "Admin Viewer" });
    renderWithProviders(<TeamGuardrailsTab accessToken="test-token" />);

    await screen.findByText("test-pending-guardrail");

    expect(screen.queryByRole("button", { name: /approve/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /reject/i })).not.toBeInTheDocument();
  });

  it("shows Approve and Reject buttons for an admin on a pending submission", async () => {
    mockUseAuthorized.mockReturnValue({ ...baseAuth, userRole: "Admin" });
    renderWithProviders(<TeamGuardrailsTab accessToken="test-token" />);

    await screen.findByText("test-pending-guardrail");

    expect(screen.getByRole("button", { name: /approve/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /reject/i })).toBeInTheDocument();
  });

  it("hides Approve and Reject buttons when the session has no role (defaults to non-admin)", async () => {
    mockUseAuthorized.mockReturnValue({ ...baseAuth, userRole: "Undefined Role" });
    renderWithProviders(<TeamGuardrailsTab accessToken="test-token" />);

    await screen.findByText("test-pending-guardrail");

    expect(screen.queryByRole("button", { name: /approve/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /reject/i })).not.toBeInTheDocument();
  });

  it("disables all admin-only write controls for a non-admin, including the detail panel", async () => {
    mockUseAuthorized.mockReturnValue({ ...baseAuth, userRole: "Internal User" });
    renderWithProviders(<TeamGuardrailsTab accessToken="test-token" />);

    await screen.findByText("test-pending-guardrail");
    expect(screen.getByRole("switch")).toBeDisabled();

    fireEvent.click(screen.getByRole("button", { name: "Review" }));
    await screen.findByText("Forward LiteLLM API Key");

    expect(screen.queryByRole("button", { name: /approve/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /reject/i })).not.toBeInTheDocument();
    screen.getAllByRole("switch").forEach((toggle) => expect(toggle).toBeDisabled());
    expect(screen.queryByRole("button", { name: "Add" })).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/^Remove/)).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText("e.g. x-request-id")).not.toBeInTheDocument();
  });

  it("keeps all write controls enabled for an admin in the detail panel", async () => {
    mockUseAuthorized.mockReturnValue({ ...baseAuth, userRole: "Admin" });
    renderWithProviders(<TeamGuardrailsTab accessToken="test-token" />);

    await screen.findByText("test-pending-guardrail");

    fireEvent.click(screen.getByRole("button", { name: "Review" }));
    await screen.findByText("Forward LiteLLM API Key");

    expect(screen.getAllByRole("button", { name: /approve/i }).length).toBeGreaterThanOrEqual(2);
    screen.getAllByRole("switch").forEach((toggle) => expect(toggle).toBeEnabled());
    expect(screen.getAllByRole("button", { name: "Add" })).toHaveLength(2);
    expect(screen.getByLabelText("Remove X-API-Key")).toBeInTheDocument();
    expect(screen.getByLabelText("Remove x-request-id")).toBeInTheDocument();
  });
});

describe("TeamGuardrailsTab URL state", () => {
  const mockUseAuthorized = vi.mocked(useAuthorized);
  const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
    onUrlUpdate.mock.calls.at(-1)?.[0];

  beforeEach(() => {
    vi.clearAllMocks();
    mockUseAuthorized.mockReturnValue({ ...baseAuth, userRole: "Admin" });
    vi.mocked(listGuardrailSubmissions).mockResolvedValue({
      submissions: [pendingSubmission],
      summary: { total: 1, pending_review: 1, active: 0, rejected: 0 },
    });
  });

  it("loads submissions with the search and status from ?sub_q= and ?sub_status=", async () => {
    renderWithProviders(<TeamGuardrailsTab accessToken="test-token" />, {
      searchParams: "?sub_q=pii&sub_status=pending",
    });

    expect(screen.getByPlaceholderText("Search guardrails...")).toHaveValue("pii");
    expect(screen.getByLabelText("Filter by status")).toHaveValue("pending");
    await waitFor(() =>
      expect(listGuardrailSubmissions).toHaveBeenCalledWith("test-token", { status: "pending_review", search: "pii" }),
    );
    expect(listGuardrailSubmissions).toHaveBeenCalledTimes(1);
  });

  it("treats an unknown ?sub_status= as all statuses", async () => {
    renderWithProviders(<TeamGuardrailsTab accessToken="test-token" />, { searchParams: "?sub_status=archived" });

    expect(screen.getByLabelText("Filter by status")).toHaveValue("all");
    await waitFor(() =>
      expect(listGuardrailSubmissions).toHaveBeenCalledWith("test-token", { status: undefined, search: undefined }),
    );
  });

  it("writes the search box to ?sub_q=", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<TeamGuardrailsTab accessToken="test-token" />, { onUrlUpdate });
    await screen.findByText("test-pending-guardrail");

    fireEvent.change(screen.getByPlaceholderText("Search guardrails..."), { target: { value: "toxic" } });

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("sub_q")).toBe("toxic"));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tab")).toBe("submitted");
    await waitFor(() =>
      expect(listGuardrailSubmissions).toHaveBeenLastCalledWith("test-token", { status: undefined, search: "toxic" }),
    );
  });

  it("writes the status filter to ?sub_status= and removes it for all statuses", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<TeamGuardrailsTab accessToken="test-token" />, { onUrlUpdate });
    await screen.findByText("test-pending-guardrail");
    const statusSelect = screen.getByLabelText("Filter by status");

    fireEvent.change(statusSelect, { target: { value: "rejected" } });
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("sub_status")).toBe("rejected"));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tab")).toBe("submitted");
    await waitFor(() =>
      expect(listGuardrailSubmissions).toHaveBeenLastCalledWith("test-token", {
        status: "rejected",
        search: undefined,
      }),
    );

    fireEvent.change(statusSelect, { target: { value: "all" } });
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("sub_status")).toBe(false));
  });

  it("opens the submission named by ?submission=", async () => {
    renderWithProviders(<TeamGuardrailsTab accessToken="test-token" />, { searchParams: "?submission=guard-1" });

    expect(await screen.findByText("Forward LiteLLM API Key")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Close detail panel" })).toBeInTheDocument();
  });

  it("pushes ?submission= when a submission is opened and replaces it away when the panel is closed", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<TeamGuardrailsTab accessToken="test-token" />, {
      searchParams: "?sub_status=pending",
      onUrlUpdate,
    });

    fireEvent.click(await screen.findByRole("button", { name: "Review" }));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("submission")).toBe("guard-1"));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tab")).toBe("submitted");
    expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("push");
    expect(await screen.findByText("Forward LiteLLM API Key")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Close detail panel" }));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("submission")).toBe(false));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("sub_status")).toBe("pending");
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tab")).toBe("submitted");
    expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("replace");
    expect(screen.queryByText("Forward LiteLLM API Key")).not.toBeInTheDocument();
  });

  it("keeps ?tab=submitted in a non-admin's submission link, whose default tab an admin does not share", async () => {
    mockUseAuthorized.mockReturnValue({ ...baseAuth, userRole: "Internal User" });
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<TeamGuardrailsTab accessToken="test-token" />, { onUrlUpdate });

    fireEvent.click(await screen.findByRole("button", { name: "Review" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("submission")).toBe("guard-1"));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tab")).toBe("submitted");
  });

  it.each([
    { action: "approve", request: approveGuardrailSubmission },
    { action: "reject", request: rejectGuardrailSubmission },
  ])("closes the open submission after it is confirmed with $action", async ({ action, request }) => {
    vi.mocked(request).mockResolvedValue({ guardrail_id: "guard-1", status: action, message: "ok" });
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<TeamGuardrailsTab accessToken="test-token" />, {
      searchParams: "?tab=submitted&submission=guard-1",
      onUrlUpdate,
    });
    await screen.findByText("Forward LiteLLM API Key");
    const actionButton = () => screen.getAllByRole("button", { name: new RegExp(`^${action}$`, "i") }).at(-1)!;

    fireEvent.click(actionButton());
    await screen.findByText(`Are you sure you want to ${action}`, { exact: false });
    fireEvent.click(actionButton());

    await waitFor(() => expect(request).toHaveBeenCalledWith("test-token", "guard-1"));
    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("submission")).toBe(false));
    expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("replace");
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("tab")).toBe("submitted");
    await waitFor(() => expect(screen.queryByText("Forward LiteLLM API Key")).not.toBeInTheDocument());
  });
});
