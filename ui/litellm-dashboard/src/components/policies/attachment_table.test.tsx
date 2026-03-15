import React from "react";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AttachmentTable from "./attachment_table";
import { PolicyAttachment } from "./types";

vi.mock("./impact_popover", () => ({
  default: () => <button aria-label="View blast radius" />,
}));

vi.mock("@heroicons/react/outline", () => ({
  TrashIcon: function TrashIcon() { return null; },
  SwitchVerticalIcon: function SwitchVerticalIcon() { return null; },
  ChevronUpIcon: function ChevronUpIcon() { return null; },
  ChevronDownIcon: function ChevronDownIcon() { return null; },
  PencilAltIcon: function PencilAltIcon() { return null; },
  PlayIcon: function PlayIcon() { return null; },
  RefreshIcon: function RefreshIcon() { return null; },
  ExternalLinkIcon: function ExternalLinkIcon() { return null; },
}));

vi.mock("@tremor/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tremor/react")>();
  return {
    ...actual,
    Button: React.forwardRef<HTMLButtonElement, any>(({ children, ...props }, ref) =>
      React.createElement("button", { ...props, ref }, children)
    ),
    Tooltip: ({ children }: { children?: React.ReactNode }) =>
      React.createElement(React.Fragment, null, children),
    Switch: ({ checked, onChange, className }: { checked?: boolean; onChange?: (v: boolean) => void; className?: string }) =>
      React.createElement("input", {
        type: "checkbox",
        role: "switch",
        checked,
        onChange: (e: React.ChangeEvent<HTMLInputElement>) => onChange?.(e.target.checked),
        className,
      }),
    Icon: ({ icon: IconComp, onClick, className }: any) =>
      React.createElement("button", { type: "button", onClick, className }, IconComp?.displayName ?? IconComp?.name ?? "icon"),
  };
});

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

  it("should render", () => {
    renderWithProviders(<AttachmentTable {...defaultProps} />);
    expect(screen.getByText("Policy")).toBeInTheDocument();
  });

  it("should show a loading message when isLoading is true", () => {
    renderWithProviders(<AttachmentTable {...defaultProps} isLoading />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it("should show 'No attachments found' when there are no attachments", () => {
    renderWithProviders(<AttachmentTable {...defaultProps} />);
    expect(screen.getByText(/no attachments found/i)).toBeInTheDocument();
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

  it("should show 'Global (*)' badge when scope is '*'", () => {
    const attachments = [makeAttachment({ scope: "*" })];
    renderWithProviders(<AttachmentTable {...defaultProps} attachments={attachments} />);
    expect(screen.getByText("Global (*)")).toBeInTheDocument();
  });

  it("should show team tags when the attachment has teams", () => {
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

  it("should call onDeleteClick with the attachment_id when the delete icon is clicked", async () => {
    const attachment = makeAttachment({ attachment_id: "att-del-me1" });
    const user = userEvent.setup();
    renderWithProviders(<AttachmentTable {...defaultProps} attachments={[attachment]} />);
    await user.click(screen.getByRole("button", { name: /TrashIcon/i }));
    expect(defaultProps.onDeleteClick).toHaveBeenCalledWith("att-del-me1");
  });

  it("should not show the delete icon for non-admins", () => {
    const attachment = makeAttachment();
    renderWithProviders(<AttachmentTable {...defaultProps} attachments={[attachment]} isAdmin={false} />);
    expect(screen.queryByRole("button", { name: /TrashIcon/i })).not.toBeInTheDocument();
  });

  it("should show a truncated attachment ID in the table", () => {
    const attachment = makeAttachment({ attachment_id: "att-abcdef1234567" });
    renderWithProviders(<AttachmentTable {...defaultProps} attachments={[attachment]} />);
    expect(screen.getByText("att-abc...")).toBeInTheDocument();
  });

  it("should render model tags when the attachment has models", () => {
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
