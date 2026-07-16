import React from "react";
import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "@/../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import PolicyTable from "./PolicyTable";
import { Policy } from "@/components/policies/types";

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
    expect(screen.getByText("Guardrails (Add)")).toBeInTheDocument();
    expect(screen.getByText("Created At")).toBeInTheDocument();
  });

  it("should show skeleton rows when isLoading is true", () => {
    renderWithProviders(<PolicyTable {...defaultProps} isLoading />);
    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
  });

  it("should show the empty state when there are no policies", () => {
    renderWithProviders(<PolicyTable {...defaultProps} />);
    expect(screen.getByText("No policies found")).toBeInTheDocument();
  });

  it("should render a clickable name cell for each grouped policy", () => {
    const policies = [
      makePolicy({ policy_name: "alpha-policy", policy_id: "id-1" }),
      makePolicy({ policy_name: "beta-policy", policy_id: "id-2" }),
    ];
    renderWithProviders(<PolicyTable {...defaultProps} policies={policies} />);
    expect(screen.getByRole("button", { name: "alpha-policy" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "beta-policy" })).toBeInTheDocument();
  });

  it("should sort rows by policy name ascending by default", () => {
    const policies = [
      makePolicy({ policy_name: "zeta-policy", policy_id: "id-z" }),
      makePolicy({ policy_name: "alpha-policy", policy_id: "id-a" }),
    ];
    renderWithProviders(<PolicyTable {...defaultProps} policies={policies} />);
    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("alpha-policy")).toBeInTheDocument();
    expect(within(rows[1]).getByText("zeta-policy")).toBeInTheDocument();
  });

  it("should call onViewClick with the policy_id when the policy name is clicked", async () => {
    const user = userEvent.setup();
    const policy = makePolicy({ policy_name: "my-policy", policy_id: "view-id-1" });
    renderWithProviders(<PolicyTable {...defaultProps} policies={[policy]} />);
    await user.click(screen.getByRole("button", { name: "my-policy" }));
    expect(defaultProps.onViewClick).toHaveBeenCalledWith("view-id-1");
  });

  it("should call onDeleteClick with policy_id and policy_name from the actions menu", async () => {
    const user = userEvent.setup();
    const policy = makePolicy({ policy_name: "del-policy", policy_id: "del-id-1" });
    renderWithProviders(<PolicyTable {...defaultProps} policies={[policy]} />);
    await user.click(screen.getByTestId("policy-actions-del-id-1"));
    await user.click(await screen.findByTestId("policy-action-delete"));
    expect(defaultProps.onDeleteClick).toHaveBeenCalledWith("del-id-1", "del-policy");
  });

  it("should call onEditClick with the policy from the actions menu", async () => {
    const user = userEvent.setup();
    const policy = makePolicy({ policy_name: "edit-policy", policy_id: "edit-id-1" });
    renderWithProviders(<PolicyTable {...defaultProps} policies={[policy]} />);
    await user.click(screen.getByTestId("policy-actions-edit-id-1"));
    await user.click(await screen.findByTestId("policy-action-edit"));
    expect(defaultProps.onEditClick).toHaveBeenCalledWith(policy);
  });

  it("should not show the actions menu for non-admins", () => {
    const policy = makePolicy();
    renderWithProviders(<PolicyTable {...defaultProps} policies={[policy]} isAdmin={false} />);
    expect(screen.queryByTestId(`policy-actions-${policy.policy_id}`)).not.toBeInTheDocument();
  });

  it("should show a version badge when multiple versions of the same policy name exist", () => {
    const publishedVersion: Partial<Policy> = {
      policy_name: "versioned",
      policy_id: "v1",
      version_status: "published",
      version_number: 1,
    };
    const productionVersion: Partial<Policy> = {
      policy_name: "versioned",
      policy_id: "v2",
      version_status: "production",
      version_number: 2,
    };
    const policies = [makePolicy(publishedVersion), makePolicy(productionVersion)];
    renderWithProviders(<PolicyTable {...defaultProps} policies={policies} />);
    expect(screen.getByText("2 versions")).toBeInTheDocument();
  });

  it("should group policies with the same name into a single row", () => {
    const policies = [
      makePolicy({ policy_name: "shared", policy_id: "s1", version_status: "published" }),
      makePolicy({ policy_name: "shared", policy_id: "s2", version_status: "production" }),
    ];
    renderWithProviders(<PolicyTable {...defaultProps} policies={policies} />);
    expect(screen.getAllByText("shared")).toHaveLength(1);
  });

  it("should show an overflow badge when more than 2 guardrails_add exist", () => {
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
    await user.click(screen.getByRole("button", { name: /grouped/ }));
    expect(defaultProps.onViewClick).toHaveBeenCalledWith("prod-id");
  });
});
