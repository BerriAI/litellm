import { renderWithProviders, screen, within } from "@/../tests/test-utils";
import type { ReactNode } from "react";
import { describe, expect, it } from "vitest";

import { PageTabs, PageTabsBar, PageTabsTrigger } from "./Page";

function renderBar(actions: ReactNode) {
  return renderWithProviders(
    <PageTabs defaultValue="budgets">
      <PageTabsBar actions={actions} aria-label="Budget views">
        <PageTabsTrigger value="budgets">Budgets</PageTabsTrigger>
        <PageTabsTrigger value="examples">Examples</PageTabsTrigger>
      </PageTabsBar>
    </PageTabs>,
  );
}

describe("PageTabsBar", () => {
  it("should keep the call to action outside the tab list", () => {
    renderBar(<button>Create Budget</button>);

    const tablist = screen.getByRole("tablist", { name: "Budget views" });
    expect(
      within(tablist)
        .getAllByRole("tab")
        .map((tab) => tab.textContent),
    ).toEqual(["Budgets", "Examples"]);
    expect(within(tablist).queryByRole("button", { name: "Create Budget" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Create Budget" })).toBeInTheDocument();
  });

  it("should separate the call to action from the tabs", () => {
    renderBar(<button>Create Budget</button>);

    expect(screen.getByRole("separator")).toHaveAttribute("aria-orientation", "vertical");
  });

  it("should render only the tabs when there is no call to action", () => {
    renderBar(false);

    expect(screen.getByRole("tablist", { name: "Budget views" })).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
    expect(screen.queryByRole("separator")).not.toBeInTheDocument();
  });
});
