import { renderWithProviders, screen, within } from "@/../tests/test-utils";
import { describe, expect, it } from "vitest";

import { PageHeader } from "./PageHeader";

const identity = {
  icon: <span>Teams icon</span>,
  title: "Teams",
  subtitle: "Manage teams, members, and their access to models and budgets",
};

describe("PageHeader", () => {
  it("should render the page identity", () => {
    renderWithProviders(<PageHeader {...identity} />);

    expect(screen.getByRole("heading", { name: "Teams" })).toBeInTheDocument();
    expect(screen.getByText("Teams icon").parentElement).toHaveAttribute("aria-hidden", "true");
    expect(screen.getByText(identity.subtitle)).toBeInTheDocument();
  });

  it("should apply the standard title and subtext typography", () => {
    renderWithProviders(<PageHeader {...identity} />);

    const icon = screen.getByText("Teams icon").parentElement;
    expect(screen.getByRole("heading", { name: "Teams" })).toHaveClass("text-2xl", "font-semibold", "tracking-tight");
    expect(screen.getByText(identity.subtitle)).toHaveClass("mt-1.5", "text-sm", "text-muted-foreground");
    expect(icon).toHaveClass("size-5", "[&_svg]:size-5", "[&_svg]:stroke-[1.75]");
    expect(icon?.parentElement).toHaveClass("gap-2.5");
  });

  it("should render the primary action, divider, tabs, and utilities in the standard control row", () => {
    renderWithProviders(
      <PageHeader
        {...identity}
        primaryAction={<button>Create Team</button>}
        tabs={
          <div role="tablist">
            <button role="tab">Your Teams</button>
          </div>
        }
        utilities={<button>Refresh</button>}
      />,
    );

    const controls = screen.getByRole("group", { name: "Page controls" });
    expect(controls).toHaveClass("mt-5", "h-9");
    expect(within(controls).getByRole("separator")).toHaveClass("mx-4", "h-6");
    expect(controls).toHaveTextContent("Create TeamYour TeamsRefresh");
  });

  it("should omit the divider when tabs are absent", () => {
    renderWithProviders(<PageHeader {...identity} primaryAction={<button>Create Team</button>} />);

    expect(screen.queryByRole("separator")).not.toBeInTheDocument();
  });

  it("should provide standard controls to an embedded tab shell", () => {
    renderWithProviders(
      <PageHeader
        {...identity}
        primaryAction={<button>Create Team</button>}
        tabs={({ leadingControls, utilities }) => (
          <div role="tablist">
            {leadingControls}
            <button role="tab">Your Teams</button>
            {utilities}
          </div>
        )}
        utilities={<button>Refresh</button>}
      />,
    );

    const tabs = screen.getByRole("tablist");
    expect(within(tabs).getByRole("separator")).toBeInTheDocument();
    expect(tabs).toHaveTextContent("Create TeamYour TeamsRefresh");
  });
});
