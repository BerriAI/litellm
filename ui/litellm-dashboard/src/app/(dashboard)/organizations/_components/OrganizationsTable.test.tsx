import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import React from "react";
import { describe, expect, it, vi, type Mock } from "vitest";

import { renderWithProviders, screen, waitFor, within } from "../../../../../tests/test-utils";

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

const thirtyOrganizations = Array.from({ length: 30 }, (_, index) =>
  makeOrganization({ organization_id: `org-${index}`, organization_alias: `Org ${index}` }),
);

const sortableOrganization = (alias: string, createdAt: string, spend: number): Organization => {
  const overrides: Partial<Organization> = {
    organization_id: `org-${alias.toLowerCase()}`,
    organization_alias: alias,
    created_at: createdAt,
    spend,
  };
  return makeOrganization(overrides);
};

const sortableOrganizations = [
  sortableOrganization("Mid", "2024-03-01T00:00:00Z", 5),
  sortableOrganization("Zed", "2023-01-01T00:00:00Z", 1),
  sortableOrganization("Ace", "2025-01-01T00:00:00Z", 3),
];

const bodyRowAliases = () =>
  screen
    .getAllByRole("row")
    .slice(1)
    .map((row) => ["Ace", "Mid", "Zed"].find((alias) => within(row).queryByText(alias) !== null));

const lastSearchParams = (onUrlUpdate: Mock<OnUrlUpdateFunction>) => onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;

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
    renderWithProviders(<OrganizationsTable {...baseProps} organizations={[]} />);
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
    renderWithProviders(
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
    renderWithProviders(
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
    renderWithProviders(
      <OrganizationsTable
        {...baseProps}
        userRole="Internal User"
        organizations={[makeOrganization({ organization_id: "org-9" })]}
      />,
    );

    expect(screen.queryByTestId("organization-actions-org-9")).not.toBeInTheDocument();
  });

  it("sorts by created_at descending by default", () => {
    renderWithProviders(
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
    renderWithProviders(
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
    renderWithProviders(
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

  it("renders a tpm/rpm limit of 0 as 0, never as Unlimited", () => {
    renderWithProviders(
      <OrganizationsTable
        {...baseProps}
        organizations={[makeOrganization({ litellm_budget_table: { max_budget: null, tpm_limit: 0, rpm_limit: 0 } })]}
      />,
    );

    expect(screen.getByText("TPM: 0")).toBeInTheDocument();
    expect(screen.getByText("RPM: 0")).toBeInTheDocument();
    expect(screen.queryByText("TPM: Unlimited")).not.toBeInTheDocument();
    expect(screen.queryByText("RPM: Unlimited")).not.toBeInTheDocument();
  });

  it("renders loading skeletons instead of rows while loading", () => {
    renderWithProviders(
      <OrganizationsTable
        {...baseProps}
        isLoading
        organizations={[makeOrganization({ organization_alias: "ShouldNotShow" })]}
      />,
    );

    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
    expect(screen.queryByText("ShouldNotShow")).not.toBeInTheDocument();
  });

  it("pages long lists client-side with the shared size selector and footer", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<OrganizationsTable {...baseProps} organizations={thirtyOrganizations} />, { onUrlUpdate });

    expect(screen.getAllByRole("row")).toHaveLength(26);
    expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-25 of 30");

    await user.click(screen.getByTestId("pagination-page-size"));
    await user.click(await screen.findByRole("option", { name: "50" }));

    expect(screen.getAllByRole("row")).toHaveLength(31);
    expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-30 of 30");
    await waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("page_size")).toBe("50"));
  });

  it("uses a search-aware empty state", () => {
    const { rerender } = renderWithProviders(
      <OrganizationsTable {...baseProps} searchActive={false} organizations={[]} />,
    );
    expect(screen.getByText("No organizations yet")).toBeInTheDocument();

    rerender(<OrganizationsTable {...baseProps} searchActive={true} organizations={[]} />);
    expect(screen.getByText("No matching organizations")).toBeInTheDocument();
  });
});

describe("OrganizationsTable URL state", () => {
  it("restores the sort column and direction from ?sort_by=&sort_order=", () => {
    renderWithProviders(<OrganizationsTable {...baseProps} organizations={sortableOrganizations} />, {
      searchParams: "?sort_by=spend&sort_order=desc",
    });

    expect(bodyRowAliases()).toEqual(["Mid", "Ace", "Zed"]);
  });

  it("falls back to sorting by creation date for a ?sort_by= column that cannot be sorted", () => {
    renderWithProviders(<OrganizationsTable {...baseProps} organizations={sortableOrganizations} />, {
      searchParams: "?sort_by=members&sort_order=asc",
    });

    expect(bodyRowAliases()).toEqual(["Zed", "Mid", "Ace"]);
  });

  it("writes the clicked sort column to the URL and returns to the first page", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<OrganizationsTable {...baseProps} organizations={thirtyOrganizations} />, {
      searchParams: "?page=2",
      onUrlUpdate,
    });
    expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 26-30 of 30");

    await user.click(screen.getByTestId("sort-header-organization_alias"));

    await waitFor(() => expect(lastSearchParams(onUrlUpdate)?.get("sort_by")).toBe("organization_alias"));
    expect(onUrlUpdate).toHaveBeenCalledTimes(1);
    expect(lastSearchParams(onUrlUpdate)?.get("sort_order")).toBe("asc");
    expect(lastSearchParams(onUrlUpdate)?.has("page")).toBe(false);
    expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-25 of 30");
    expect(within(screen.getAllByRole("row")[1]).getByText("Org 0")).toBeInTheDocument();
  });

  it("opens the page named by ?page= and writes page changes back to the URL", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<OrganizationsTable {...baseProps} organizations={thirtyOrganizations} />, {
      searchParams: "?page=2",
      onUrlUpdate,
    });

    expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 26-30 of 30");
    expect(screen.getByText("org-29")).toBeInTheDocument();

    await user.click(screen.getByTestId("pagination-prev"));

    await waitFor(() => expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 1-25 of 30"));
    expect(lastSearchParams(onUrlUpdate)?.has("page")).toBe(false);

    await user.click(screen.getByTestId("pagination-next"));

    await waitFor(() => expect(lastSearchParams(onUrlUpdate)?.get("page")).toBe("2"));
  });

  it("keeps a deep-linked ?page= while the organization list is still loading", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    const { rerender } = renderWithProviders(<OrganizationsTable {...baseProps} isLoading organizations={[]} />, {
      searchParams: "?page=2",
      onUrlUpdate,
    });

    rerender(<OrganizationsTable {...baseProps} organizations={thirtyOrganizations} />);

    await waitFor(() => expect(screen.getByTestId("pagination-range")).toHaveTextContent("Showing 26-30 of 30"));
    expect(onUrlUpdate).not.toHaveBeenCalled();
  });
});
