import { renderWithProviders, screen, within } from "@/../tests/test-utils";
import { Users } from "lucide-react";
import { describe, expect, it } from "vitest";

import { PageHeader, PageHeaderControls, PageHeaderDescription, PageHeaderTitle } from "./PageHeader";

describe("PageHeader", () => {
  it("should name the page heading by its text alone when it carries an icon", () => {
    renderWithProviders(
      <PageHeader>
        <PageHeaderTitle>
          <Users />
          Teams
        </PageHeaderTitle>
        <PageHeaderDescription>Manage teams</PageHeaderDescription>
      </PageHeader>,
    );

    expect(screen.getByRole("heading", { level: 1, name: "Teams" })).toBeInTheDocument();
    expect(screen.getByText("Manage teams")).toBeInTheDocument();
  });

  it("should group the composed controls under one accessible label", () => {
    renderWithProviders(
      <PageHeader>
        <PageHeaderTitle>Teams</PageHeaderTitle>
        <PageHeaderControls>
          <button>Create Team</button>
          <button>Refresh</button>
        </PageHeaderControls>
      </PageHeader>,
    );

    const controls = screen.getByRole("group", { name: "Page controls" });
    expect(
      within(controls)
        .getAllByRole("button")
        .map((button) => button.textContent),
    ).toEqual(["Create Team", "Refresh"]);
  });

  it("should forward native attributes to each part", () => {
    renderWithProviders(
      <PageHeader data-testid="header">
        <PageHeaderTitle id="page-title">Teams</PageHeaderTitle>
        <PageHeaderControls aria-label="Team controls" />
      </PageHeader>,
    );

    expect(screen.getByTestId("header")).toContainElement(screen.getByRole("heading", { name: "Teams" }));
    expect(screen.getByRole("heading", { name: "Teams" })).toHaveAttribute("id", "page-title");
    expect(screen.getByRole("group", { name: "Team controls" })).toBeInTheDocument();
  });
});
