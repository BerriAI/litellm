import React from "react";
import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "@/../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "@/components/networking";
import AddAttachmentForm from "./add_attachment_form";
import { Policy } from "@/components/policies/types";

vi.mock("@/components/networking");

vi.mock("@/components/molecules/notifications_manager", () => ({
  default: { success: vi.fn(), fromBackend: vi.fn(), error: vi.fn(), info: vi.fn() },
}));

vi.mock("./impact_preview_alert", () => ({
  default: ({ impactResult }: { impactResult: any }) =>
    React.createElement("div", { "data-testid": "impact-preview" }, `${impactResult.affected_keys_count} keys`),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ userId: "admin-user-id", userRole: "Admin", accessToken: "test-token" }),
}));

const makePolicy = (overrides: Partial<Policy> = {}): Policy => ({
  policy_id: "policy-id-1",
  policy_name: "test-policy",
  inherit: null,
  description: null,
  guardrails_add: [],
  guardrails_remove: [],
  condition: null,
  ...overrides,
});

const defaultProps = {
  visible: true,
  onClose: vi.fn(),
  onSuccess: vi.fn(),
  accessToken: "test-token",
  policies: [
    makePolicy({ policy_name: "policy-alpha" }),
    makePolicy({ policy_name: "policy-beta", policy_id: "id-2" }),
  ],
  createAttachment: vi.fn(),
};

const teamListResult = (aliases: string[]) =>
  aliases.map((team_alias) => ({ team_alias })) as unknown as Awaited<ReturnType<typeof networking.teamListCall>>;

describe("AddAttachmentForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(networking.teamListCall).mockResolvedValue([]);
    vi.mocked(networking.keyListCall).mockResolvedValue({ keys: [] });
    vi.mocked(networking.modelAvailableCall).mockResolvedValue({ data: [] });
  });

  it("should render the modal title when visible", async () => {
    renderWithProviders(<AddAttachmentForm {...defaultProps} />);
    expect(await screen.findByText("Create Policy Attachment")).toBeInTheDocument();
  });

  it("should not render modal content when visible is false", () => {
    renderWithProviders(<AddAttachmentForm {...defaultProps} visible={false} />);
    expect(screen.queryByText("Create Policy Attachment")).not.toBeInTheDocument();
  });

  it("should fetch teams, keys, and models on mount when visible and accessToken are provided", async () => {
    renderWithProviders(<AddAttachmentForm {...defaultProps} />);
    await waitFor(() => {
      expect(networking.teamListCall).toHaveBeenCalled();
      expect(networking.keyListCall).toHaveBeenCalled();
      expect(networking.modelAvailableCall).toHaveBeenCalled();
    });
  });

  it("fetches all teams, not just teams the caller is a member of (LIT-4199)", async () => {
    renderWithProviders(<AddAttachmentForm {...defaultProps} />);
    await waitFor(() => expect(networking.teamListCall).toHaveBeenCalled());
    expect(networking.teamListCall).toHaveBeenCalledWith("test-token", null, null);
  });

  it("should not fetch teams, keys, or models when accessToken is null", () => {
    renderWithProviders(<AddAttachmentForm {...defaultProps} accessToken={null} />);
    expect(networking.teamListCall).not.toHaveBeenCalled();
    expect(networking.keyListCall).not.toHaveBeenCalled();
    expect(networking.modelAvailableCall).not.toHaveBeenCalled();
  });

  it("should call onClose when the Cancel button is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AddAttachmentForm {...defaultProps} />);
    await user.click(await screen.findByRole("button", { name: /cancel/i }));
    expect(defaultProps.onClose).toHaveBeenCalled();
  });

  it("should not show scope-specific fields when scope is global (default)", async () => {
    renderWithProviders(<AddAttachmentForm {...defaultProps} />);
    await screen.findByText("Create Policy Attachment");
    expect(screen.queryByText("Teams")).not.toBeInTheDocument();
    expect(screen.queryByText("Keys")).not.toBeInTheDocument();
    expect(screen.queryByText("Models")).not.toBeInTheDocument();
  });

  it("should show Teams, Keys, Models, and Tags fields when scope is switched to specific", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AddAttachmentForm {...defaultProps} />);
    await screen.findByText("Create Policy Attachment");
    await user.click(screen.getByRole("radio", { name: /specific/i }));
    expect(screen.getByText("Teams")).toBeInTheDocument();
    expect(screen.getByText("Keys")).toBeInTheDocument();
    expect(screen.getByText("Models")).toBeInTheDocument();
    expect(screen.getByText("Tags")).toBeInTheDocument();
  });

  it("should show the 'Estimate Impact' button only when scope is specific", async () => {
    const user = userEvent.setup();
    renderWithProviders(<AddAttachmentForm {...defaultProps} />);
    await screen.findByText("Create Policy Attachment");
    expect(screen.queryByRole("button", { name: /estimate impact/i })).not.toBeInTheDocument();
    await user.click(screen.getByRole("radio", { name: /specific/i }));
    expect(screen.getByRole("button", { name: /estimate impact/i })).toBeInTheDocument();
  });

  it("should render a 'Create Attachment' submit button", async () => {
    renderWithProviders(<AddAttachmentForm {...defaultProps} />);
    expect(await screen.findByRole("button", { name: /create attachment/i })).toBeInTheDocument();
  });

  type UserEvent = ReturnType<typeof userEvent.setup>;

  const TEAMS_ERROR = /these teams don't exist/i;

  const openSpecificScope = async (user: UserEvent) => {
    await screen.findByText("Create Policy Attachment");
    await waitFor(() => expect(networking.teamListCall).toHaveBeenCalled());
    await user.click(screen.getByRole("radio", { name: /specific/i }));
  };

  const enterTeam = async (user: UserEvent, value: string) => {
    const item = screen.getByText("Teams").closest(".ant-form-item") as HTMLElement;
    const input = within(item).getByRole("combobox");
    await user.click(input);
    await user.type(input, `${value}{Enter}`);
  };

  // Submits and waits for the validation cycle to settle. No policy is selected, so
  // the "select at least one policy" required error always appears - we use it as a
  // synchronization point, then assert whether the teams validator also complained.
  const submitAndSettle = async (user: UserEvent) => {
    await user.click(screen.getByRole("button", { name: /create attachment/i }));
    await screen.findByText(/select at least one policy/i);
  };

  it("blocks submit with a field error when a concrete team that does not exist is entered", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.teamListCall).mockResolvedValue(teamListResult(["real-team"]));
    const createAttachment = vi.fn();
    renderWithProviders(<AddAttachmentForm {...defaultProps} createAttachment={createAttachment} />);
    await openSpecificScope(user);
    await enterTeam(user, "ghost-team");
    await submitAndSettle(user);
    expect(screen.getByText(TEAMS_ERROR)).toBeInTheDocument();
    expect(createAttachment).not.toHaveBeenCalled();
  });

  it("does not flag a team that exists", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.teamListCall).mockResolvedValue(teamListResult(["real-team"]));
    renderWithProviders(<AddAttachmentForm {...defaultProps} />);
    await openSpecificScope(user);
    await enterTeam(user, "real-team");
    await submitAndSettle(user);
    expect(screen.queryByText(TEAMS_ERROR)).not.toBeInTheDocument();
  });

  it("does not flag a wildcard pattern even when it matches no existing team", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.teamListCall).mockResolvedValue(teamListResult([]));
    renderWithProviders(<AddAttachmentForm {...defaultProps} />);
    await openSpecificScope(user);
    await enterTeam(user, "healthcare-*");
    await submitAndSettle(user);
    expect(screen.queryByText(TEAMS_ERROR)).not.toBeInTheDocument();
  });

  it("defers to the backend (does not flag) when the team list failed to load", async () => {
    const user = userEvent.setup();
    vi.mocked(networking.teamListCall).mockRejectedValue(new Error("boom"));
    renderWithProviders(<AddAttachmentForm {...defaultProps} />);
    await openSpecificScope(user);
    await enterTeam(user, "ghost-team");
    await submitAndSettle(user);
    expect(screen.queryByText(TEAMS_ERROR)).not.toBeInTheDocument();
  });
});
