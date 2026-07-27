import React from "react";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "@/../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AttachmentTable from "./AttachmentTable";
import { PolicyAttachment } from "@/components/policies/types";

vi.mock("./impact_popover", () => ({
  default: function ImpactPopoverMock() {
    return <button aria-label="View blast radius" />;
  },
}));

const makeAttachment = (overrides: Partial<PolicyAttachment> = {}): PolicyAttachment => ({
  attachment_id: "att-abcdef1",
  policy_name: "my-policy",
  scope: null,
  teams: [],
  keys: [],
  models: [],
  tags: [],
  ...overrides,
});

const defaultProps = {
  attachments: [],
  isLoading: false,
  onDeleteClick: vi.fn(),
  isAdmin: true,
  accessToken: "test-token",
};

describe("AttachmentTable", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render column headers", () => {
    renderWithProviders(<AttachmentTable {...defaultProps} />);
    expect(screen.getByText("Attachment ID")).toBeInTheDocument();
    expect(screen.getByText("Policy")).toBeInTheDocument();
    expect(screen.getByText("Scope")).toBeInTheDocument();
    expect(screen.getByText("Teams")).toBeInTheDocument();
    expect(screen.getByText("Keys")).toBeInTheDocument();
    expect(screen.getByText("Models")).toBeInTheDocument();
    expect(screen.getByText("Tags")).toBeInTheDocument();
    expect(screen.getByText("Created At")).toBeInTheDocument();
  });

  it("should show skeleton rows when isLoading is true", () => {
    renderWithProviders(<AttachmentTable {...defaultProps} isLoading />);
    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
  });

  it("should show the empty state when there are no attachments", () => {
    renderWithProviders(<AttachmentTable {...defaultProps} />);
    expect(screen.getByText("No attachments found")).toBeInTheDocument();
  });

  it("should render a row for each attachment", () => {
    const attachments = [
      makeAttachment({ attachment_id: "att-aaa0001", policy_name: "policy-alpha" }),
      makeAttachment({ attachment_id: "att-bbb0002", policy_name: "policy-beta" }),
    ];
    renderWithProviders(<AttachmentTable {...defaultProps} attachments={attachments} />);
    expect(screen.getByText("policy-alpha")).toBeInTheDocument();
    expect(screen.getByText("policy-beta")).toBeInTheDocument();
  });

  it("should sort rows by created_at descending by default", () => {
    const attachments = [
      makeAttachment({ attachment_id: "att-old0001", policy_name: "older-policy", created_at: "2024-01-01T00:00:00Z" }),
      makeAttachment({ attachment_id: "att-new0001", policy_name: "newer-policy", created_at: "2025-06-01T00:00:00Z" }),
    ];
    renderWithProviders(<AttachmentTable {...defaultProps} attachments={attachments} />);
    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("newer-policy")).toBeInTheDocument();
    expect(within(rows[1]).getByText("older-policy")).toBeInTheDocument();
  });

  it("should show 'Global (*)' badge when scope is '*'", () => {
    const attachments = [makeAttachment({ scope: "*" })];
    renderWithProviders(<AttachmentTable {...defaultProps} attachments={attachments} />);
    expect(screen.getByText("Global (*)")).toBeInTheDocument();
  });

  it("should show team chips when the attachment has teams", () => {
    const attachments = [makeAttachment({ teams: ["team-alpha", "team-beta"] })];
    renderWithProviders(<AttachmentTable {...defaultProps} attachments={attachments} />);
    expect(screen.getByText("team-alpha")).toBeInTheDocument();
    expect(screen.getByText("team-beta")).toBeInTheDocument();
  });

  it("should show an overflow indicator when there are more than 2 teams", () => {
    const attachments = [makeAttachment({ teams: ["t1", "t2", "t3", "t4"] })];
    renderWithProviders(<AttachmentTable {...defaultProps} attachments={attachments} />);
    expect(screen.getByText("+2")).toBeInTheDocument();
  });

  it("should call onDeleteClick with the attachment_id from the actions menu", async () => {
    const attachment = makeAttachment({ attachment_id: "att-del-me1" });
    const user = userEvent.setup();
    renderWithProviders(<AttachmentTable {...defaultProps} attachments={[attachment]} />);
    await user.click(screen.getByTestId("attachment-actions-att-del-me1"));
    await user.click(await screen.findByTestId("attachment-action-delete"));
    expect(defaultProps.onDeleteClick).toHaveBeenCalledWith("att-del-me1");
  });

  it("should not show the delete item for non-admins", async () => {
    const attachment = makeAttachment({ attachment_id: "att-nonadmin" });
    const user = userEvent.setup();
    renderWithProviders(<AttachmentTable {...defaultProps} attachments={[attachment]} isAdmin={false} />);
    await user.click(screen.getByTestId("attachment-actions-att-nonadmin"));
    expect(await screen.findByTestId("attachment-action-copy-id")).toBeInTheDocument();
    expect(screen.queryByTestId("attachment-action-delete")).not.toBeInTheDocument();
  });

  it("should copy the attachment id from the actions menu", async () => {
    const attachment = makeAttachment({ attachment_id: "att-copy-me1" });
    const user = userEvent.setup();
    renderWithProviders(<AttachmentTable {...defaultProps} attachments={[attachment]} />);
    await user.click(screen.getByTestId("attachment-actions-att-copy-me1"));
    await user.click(await screen.findByTestId("attachment-action-copy-id"));
    expect(await window.navigator.clipboard.readText()).toBe("att-copy-me1");
  });

  it("should show the blast radius action for non-admins", () => {
    const attachment = makeAttachment();
    renderWithProviders(<AttachmentTable {...defaultProps} attachments={[attachment]} isAdmin={false} />);
    expect(screen.getByRole("button", { name: "View blast radius" })).toBeInTheDocument();
  });

  it("should show the attachment ID as truncated plain mono text", () => {
    const attachment = makeAttachment({ attachment_id: "att-abcdef1234567" });
    renderWithProviders(<AttachmentTable {...defaultProps} attachments={[attachment]} />);
    const idElement = screen.getByText("att-abcdef1234567");
    expect(idElement.className).toContain("font-mono");
    expect(idElement.className).toContain("truncate");
    expect(idElement.className).not.toContain("bg-blue-50");
  });

  it("should render model chips when the attachment has models", () => {
    const attachments = [makeAttachment({ models: ["gpt-4", "claude-3"] })];
    renderWithProviders(<AttachmentTable {...defaultProps} attachments={attachments} />);
    expect(screen.getByText("gpt-4")).toBeInTheDocument();
    expect(screen.getByText("claude-3")).toBeInTheDocument();
  });

  it("should render tag chips when the attachment has tags", () => {
    const attachments = [makeAttachment({ tags: ["prod"] })];
    renderWithProviders(<AttachmentTable {...defaultProps} attachments={attachments} />);
    expect(screen.getByText("prod")).toBeInTheDocument();
  });
});
