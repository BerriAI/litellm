import React from "react";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import PolicyTable from "./policy_table";
import { Policy } from "./types";

vi.mock("@heroicons/react/outline", () => ({
  TrashIcon: function TrashIcon() { return null; },
  PencilIcon: function PencilIcon() { return null; },
  SwitchVerticalIcon: function SwitchVerticalIcon() { return null; },
  ChevronUpIcon: function ChevronUpIcon() { return null; },
  ChevronDownIcon: function ChevronDownIcon() { return null; },
}));

vi.mock("@tremor/react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tremor/react")>();
  return {
    ...actual,
    Button: React.forwardRef<HTMLButtonElement, any>(({ children, ...props }, ref) =>
      React.createElement("button", { ...props, ref }, children)
    ),
    Icon: ({ icon: IconComp, onClick, className }: any) =>
      React.createElement("button", { type: "button", onClick, className }, IconComp?.displayName ?? IconComp?.name ?? "icon"),
    Tooltip: ({ children }: { children?: React.ReactNode }) =>
      React.createElement(React.Fragment, null, children),
    Badge: ({ children }: { children?: React.ReactNode }) =>
      React.createElement("span", null, children),
  };
});

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
  policies: [],
  isLoading: false,
  onDeleteClick: vi.fn(),
  onEditClick: vi.fn(),
  onViewClick: vi.fn(),
  isAdmin: true,
};

describe("PolicyTable", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render column headers", () => {
    renderWithProviders(<PolicyTable {...defaultProps} />);
    expect(screen.getByText("Name")).toBeInTheDocument();
    expect(screen.getByText("Description")).toBeInTheDocument();
    expect(screen.getByText("Actions")).toBeInTheDocument();
  });

  it("should show a loading message when isLoading is true", () => {
    renderWithProviders(<PolicyTable {...defaultProps} isLoading />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it("should show 'No policies found' when there are no policies", () => {
    renderWithProviders(<PolicyTable {...defaultProps} />);
    expect(screen.getByText(/no policies found/i)).toBeInTheDocument();
  });

  it("should render a button with the policy name for each grouped policy", () => {
    const policies = [
      makePolicy({ policy_name: "alpha-policy", policy_id: "id-1" }),
      makePolicy({ policy_name: "beta-policy", policy_id: "id-2" }),
    ];
    renderWithProviders(<PolicyTable {...defaultProps} policies={policies} />);
    expect(screen.getByRole("button", { name: "alpha-policy" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "beta-policy" })).toBeInTheDocument();
  });

  it("should call onViewClick with the policy_id when the policy name button is clicked", async () => {
    const user = userEvent.setup();
    const policy = makePolicy({ policy_name: "my-policy", policy_id: "view-id-1" });
    renderWithProviders(<PolicyTable {...defaultProps} policies={[policy]} />);
    await user.click(screen.getByRole("button", { name: "my-policy" }));
    expect(defaultProps.onViewClick).toHaveBeenCalledWith("view-id-1");
  });

  it("should call onDeleteClick with policy_id and policy_name when the delete icon is clicked", async () => {
    const user = userEvent.setup();
    const policy = makePolicy({ policy_name: "del-policy", policy_id: "del-id-1" });
    renderWithProviders(<PolicyTable {...defaultProps} policies={[policy]} />);
    await user.click(screen.getByRole("button", { name: /TrashIcon/i }));
    expect(defaultProps.onDeleteClick).toHaveBeenCalledWith("del-id-1", "del-policy");
  });

  it("should call onEditClick with the policy when the edit icon is clicked", async () => {
    const user = userEvent.setup();
    const policy = makePolicy({ policy_name: "edit-policy", policy_id: "edit-id-1" });
    renderWithProviders(<PolicyTable {...defaultProps} policies={[policy]} />);
    await user.click(screen.getByRole("button", { name: /PencilIcon/i }));
    expect(defaultProps.onEditClick).toHaveBeenCalledWith(policy);
  });

  it("should not show admin action icons for non-admins", () => {
    const policy = makePolicy();
    renderWithProviders(<PolicyTable {...defaultProps} policies={[policy]} isAdmin={false} />);
    expect(screen.queryByRole("button", { name: /TrashIcon/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /PencilIcon/i })).not.toBeInTheDocument();
  });

  it("should show a version badge when multiple versions of the same policy name exist", () => {
    const policies = [
      makePolicy({ policy_name: "versioned", policy_id: "v1", version_status: "published", version_number: 1 }),
      makePolicy({ policy_name: "versioned", policy_id: "v2", version_status: "production", version_number: 2 }),
    ];
    renderWithProviders(<PolicyTable {...defaultProps} policies={policies} />);
    expect(screen.getByText(/2 version/i)).toBeInTheDocument();
  });

  it("should group policies with the same name into a single row", () => {
    const policies = [
      makePolicy({ policy_name: "shared", policy_id: "s1", version_status: "published" }),
      makePolicy({ policy_name: "shared", policy_id: "s2", version_status: "production" }),
    ];
    renderWithProviders(<PolicyTable {...defaultProps} policies={policies} />);
    expect(screen.getAllByRole("button", { name: "shared" })).toHaveLength(1);
  });

  it("should show an overflow tag when more than 2 guardrails_add exist", () => {
    const policy = makePolicy({ guardrails_add: ["g1", "g2", "g3", "g4"] });
    renderWithProviders(<PolicyTable {...defaultProps} policies={[policy]} />);
    expect(screen.getByText("+2")).toBeInTheDocument();
  });

  it("should prefer the production version as the primary policy when grouping", async () => {
    const user = userEvent.setup();
    const policies = [
      makePolicy({ policy_name: "grouped", policy_id: "published-id", version_status: "published" }),
      makePolicy({ policy_name: "grouped", policy_id: "prod-id", version_status: "production" }),
    ];
    renderWithProviders(<PolicyTable {...defaultProps} policies={policies} />);
    await user.click(screen.getByRole("button", { name: "grouped" }));
    expect(defaultProps.onViewClick).toHaveBeenCalledWith("prod-id");
  });
});
