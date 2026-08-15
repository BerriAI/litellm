import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, expect, it, vi } from "vitest";

import { Organization } from "@/components/networking";

import OrganizationsTable from "./OrganizationsTable";

const makeOrganization = (overrides: Partial<Organization> = {}): Organization => ({
  organization_id: "org-alpha",
  organization_alias: "Alpha",
  budget_id: "budget-1",
  metadata: {},
  models: [],
  spend: 0,
  model_spend: {},
  created_at: "2023-01-01T00:00:00Z",
  created_by: "someone",
  updated_at: "2023-01-01T00:00:00Z",
  updated_by: "someone",
  litellm_budget_table: null,
  teams: null,
  users: null,
  members: null,
  ...overrides,
});

const baseProps = {
  isLoading: false,
  userRole: "Admin",
  searchActive: false,
  onOrganizationClick: vi.fn(),
  onEditClick: vi.fn(),
  onDeleteClick: vi.fn(),
};

describe("OrganizationsTable", () => {
  it("renders every column header", () => {
    render(<OrganizationsTable {...baseProps} organizations={[]} />);
    for (const header of [
      "Organization ID",
      "Organization Name",
      "Created",
      "Spend (USD)",
      "Budget (USD)",
      "Models",
      "TPM / RPM Limits",
      "Members",
    ]) {
      expect(screen.getByText(header)).toBeInTheDocument();
    }
  });

  it("opens the detail view when the organization ID cell is clicked", async () => {
    const user = userEvent.setup();
    const onOrganizationClick = vi.fn();
    render(
      <OrganizationsTable
        {...baseProps}
        onOrganizationClick={onOrganizationClick}
        organizations={[makeOrganization({ organization_id: "org-123" })]}
      />,
    );

    await user.click(screen.getByText("org-123"));

    expect(onOrganizationClick).toHaveBeenCalledWith("org-123");
  });

  it("edits and deletes an organization through the ⋯ actions menu (admin)", async () => {
    const user = userEvent.setup();
    const onEditClick = vi.fn();
    const onDeleteClick = vi.fn();
    render(
      <OrganizationsTable
        {...baseProps}
        userRole="Admin"
        onEditClick={onEditClick}
        onDeleteClick={onDeleteClick}
        organizations={[makeOrganization({ organization_id: "org-9" })]}
      />,
    );

    await user.click(screen.getByTestId("organization-actions-org-9"));
    await user.click(await screen.findByTestId("organization-action-edit"));
    expect(onEditClick).toHaveBeenCalledWith("org-9");

    await user.click(screen.getByTestId("organization-actions-org-9"));
    await user.click(await screen.findByTestId("organization-action-delete"));
    expect(onDeleteClick).toHaveBeenCalledWith("org-9");
  });

  it("hides the row actions menu from non-admins", () => {
    render(
      <OrganizationsTable
        {...baseProps}
        userRole="Internal User"
        organizations={[makeOrganization({ organization_id: "org-9" })]}
      />,
    );

    expect(screen.queryByTestId("organization-actions-org-9")).not.toBeInTheDocument();
  });

  it("sorts by created_at descending by default", () => {
    render(
      <OrganizationsTable
        {...baseProps}
        organizations={[
          makeOrganization({
            organization_id: "org-old",
            organization_alias: "Older",
            created_at: "2023-01-01T00:00:00Z",
          }),
          makeOrganization({
            organization_id: "org-new",
            organization_alias: "Newer",
            created_at: "2024-06-01T00:00:00Z",
          }),
        ]}
      />,
    );

    const rows = screen.getAllByRole("row");
    // rows[0] is the header row; the newest organization must lead the body.
    expect(within(rows[1]).getByText("Newer")).toBeInTheDocument();
    expect(within(rows[2]).getByText("Older")).toBeInTheDocument();
  });

  it("renders budget, limits, members, and models for a fully-populated organization", () => {
    render(
      <OrganizationsTable
        {...baseProps}
        organizations={[
          makeOrganization({
            litellm_budget_table: { max_budget: 100, tpm_limit: 1000, rpm_limit: 60 },
            members: [{ user_id: "a" }, { user_id: "b" }, { user_id: "c" }],
            models: ["gpt-4o", "claude-sonnet-4", "gemini-2.5-pro", "llama-3", "mistral-large"],
          }),
        ]}
      />,
    );

    expect(screen.getByText("$100.00")).toBeInTheDocument();
    expect(screen.getByText("TPM: 1000")).toBeInTheDocument();
    expect(screen.getByText("RPM: 60")).toBeInTheDocument();
    expect(screen.getByText("3 Members")).toBeInTheDocument();
    // Five models, three visible -> the shared ModelsCell collapses the rest.
    expect(screen.getByText("+2 more")).toBeInTheDocument();
  });

  it("shows Unlimited budget and All Proxy Models when unset", () => {
    render(
      <OrganizationsTable
        {...baseProps}
        organizations={[makeOrganization({ organization_id: "org-empty", litellm_budget_table: {}, models: [] })]}
      />,
    );

    expect(screen.getByText("All Proxy Models")).toBeInTheDocument();
    // Budget shows a standalone "Unlimited"; the limits fall back inline.
    expect(screen.getByText("Unlimited")).toBeInTheDocument();
    expect(screen.getByText("TPM: Unlimited")).toBeInTheDocument();
    expect(screen.getByText("RPM: Unlimited")).toBeInTheDocument();
  });

  it("renders loading skeletons instead of rows while loading", () => {
    render(
      <OrganizationsTable
        {...baseProps}
        isLoading
        organizations={[makeOrganization({ organization_alias: "ShouldNotShow" })]}
      />,
    );

    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
    expect(screen.queryByText("ShouldNotShow")).not.toBeInTheDocument();
  });

  it("uses a search-aware empty state", () => {
    const { rerender } = render(<OrganizationsTable {...baseProps} searchActive={false} organizations={[]} />);
    expect(screen.getByText("No organizations yet")).toBeInTheDocument();

    rerender(<OrganizationsTable {...baseProps} searchActive={true} organizations={[]} />);
    expect(screen.getByText("No matching organizations")).toBeInTheDocument();
  });
});
