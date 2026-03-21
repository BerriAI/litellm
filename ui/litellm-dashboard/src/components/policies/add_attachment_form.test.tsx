import React from "react";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as networking from "../networking";
import AddAttachmentForm from "./add_attachment_form";
import { Policy } from "./types";

vi.mock("../networking");

vi.mock("./impact_preview_alert", () => ({
  default: ({ impactResult }: { impactResult: any }) =>
    React.createElement("div", { "data-testid": "impact-preview" }, `${impactResult.affected_keys_count} keys`),
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
  policies: [makePolicy({ policy_name: "policy-alpha" }), makePolicy({ policy_name: "policy-beta", policy_id: "id-2" })],
  createAttachment: vi.fn(),
};

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
});
