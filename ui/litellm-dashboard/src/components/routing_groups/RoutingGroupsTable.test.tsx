import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "../../../tests/test-utils";

import RoutingGroupsTable from "./RoutingGroupsTable";
import type { RoutingGroup } from "./types";

describe("RoutingGroupsTable", () => {
  const onEdit = vi.fn();
  const onDelete = vi.fn();

  const prodGroup: RoutingGroup = {
    group_name: "prod-group",
    models: ["gpt-4o", "claude-sonnet-4-5"],
    routing_strategy: "usage-based-routing",
  };

  const devGroup: RoutingGroup = {
    group_name: "dev-group",
    models: ["gpt-4o-mini"],
    routing_strategy: "simple-shuffle",
  };

  const manyGroups: RoutingGroup[] = Array.from({ length: 30 }, (_, index) => ({
    group_name: `group-${String(index).padStart(2, "0")}`,
    models: ["gpt-4o"],
    routing_strategy: "simple-shuffle",
  }));
  const groupNames = manyGroups.map((group) => group.group_name);

  const defaultProps = {
    groups: [] as RoutingGroup[],
    onEdit,
    onDelete,
    proxyBaseUrl: "https://proxy.example.com",
  };

  const rowFor = (groupName: string): HTMLElement => {
    const row = document.querySelector(`[data-row-id="${groupName}"]`);
    if (!(row instanceof HTMLElement)) {
      throw new Error(`No row rendered for ${groupName}`);
    }
    return row;
  };

  const namesInOrder = () =>
    screen
      .getAllByRole("row")
      .slice(1)
      .map((row) => row.getAttribute("data-row-id"))
      .filter((id): id is string => id !== null);

  const lastUrl = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
    onUrlUpdate.mock.calls.at(-1)?.[0].searchParams;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render every column header", () => {
    renderWithProviders(<RoutingGroupsTable {...defaultProps} />);
    for (const header of ["Group Name", "Models", "Strategy"]) {
      expect(screen.getByText(header)).toBeInTheDocument();
    }
  });

  it("should show the empty state when there are no groups", () => {
    renderWithProviders(<RoutingGroupsTable {...defaultProps} />);
    expect(screen.getByText("No routing groups yet")).toBeInTheDocument();
  });

  it("should render the group name, its models, and a human-readable strategy label", () => {
    renderWithProviders(<RoutingGroupsTable {...defaultProps} groups={[prodGroup]} />);
    const row = rowFor("prod-group");
    expect(within(row).getByText("prod-group")).toBeInTheDocument();
    expect(within(row).getByText("gpt-4o")).toBeInTheDocument();
    expect(within(row).getByText("claude-sonnet-4-5")).toBeInTheDocument();
    expect(within(row).getByText("Usage Based")).toBeInTheDocument();
  });

  it("should fall back to the raw strategy value when it has no friendly label", () => {
    renderWithProviders(
      <RoutingGroupsTable {...defaultProps} groups={[{ ...prodGroup, routing_strategy: "custom-strategy" }]} />,
    );
    expect(within(rowFor("prod-group")).getByText("custom-strategy")).toBeInTheDocument();
  });

  it("should collapse models beyond the first three behind a +N more badge", () => {
    const wideGroup: RoutingGroup = { ...prodGroup, models: ["a", "b", "c", "d", "e"] };
    renderWithProviders(<RoutingGroupsTable {...defaultProps} groups={[wideGroup]} />);
    const row = rowFor("prod-group");
    expect(within(row).getByText("+2 more")).toBeInTheDocument();
    expect(within(row).queryByText("d")).not.toBeInTheDocument();
  });

  it("should keep the incoming order until a column is sorted", async () => {
    const user = userEvent.setup();
    renderWithProviders(<RoutingGroupsTable {...defaultProps} groups={[prodGroup, devGroup]} />);

    expect(namesInOrder()).toEqual(["prod-group", "dev-group"]);

    await user.click(screen.getByTestId("sort-header-group_name"));
    expect(namesInOrder()).toEqual(["dev-group", "prod-group"]);
  });

  it("should toggle the usage panel when the group name is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<RoutingGroupsTable {...defaultProps} groups={[prodGroup]} />);

    expect(screen.queryByText("How routing works for this group")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "prod-group" }));
    expect(await screen.findByText("How routing works for this group")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "prod-group" }));
    expect(screen.queryByText("How routing works for this group")).not.toBeInTheDocument();
  });

  it("should build the usage snippet from the proxy base url and the group's first model", async () => {
    const user = userEvent.setup();
    renderWithProviders(<RoutingGroupsTable {...defaultProps} groups={[prodGroup]} />);
    await user.click(screen.getByRole("button", { name: "prod-group" }));

    const panel = (await screen.findByText("How routing works for this group")).closest("div")?.parentElement;
    expect(panel?.textContent).toContain("https://proxy.example.com");
    expect(panel?.textContent).toContain("gpt-4o");
  });

  it("should expand only the clicked group", async () => {
    const user = userEvent.setup();
    renderWithProviders(<RoutingGroupsTable {...defaultProps} groups={[prodGroup, devGroup]} />);

    await user.click(screen.getByRole("button", { name: "dev-group" }));
    expect(await screen.findAllByText("How routing works for this group")).toHaveLength(1);
    expect(within(rowFor("prod-group")).queryByText("How routing works for this group")).not.toBeInTheDocument();
  });

  it("should edit a group through the actions menu", async () => {
    const user = userEvent.setup();
    renderWithProviders(<RoutingGroupsTable {...defaultProps} groups={[prodGroup]} />);
    await user.click(screen.getByTestId("routing-group-actions-prod-group"));
    await user.click(await screen.findByTestId("routing-group-action-edit"));
    expect(onEdit).toHaveBeenCalledWith(prodGroup);
  });

  it("should delete a group through the actions menu", async () => {
    const user = userEvent.setup();
    renderWithProviders(<RoutingGroupsTable {...defaultProps} groups={[prodGroup]} />);
    await user.click(screen.getByTestId("routing-group-actions-prod-group"));
    await user.click(await screen.findByTestId("routing-group-action-delete"));
    expect(onDelete).toHaveBeenCalledWith(prodGroup);
  });

  it("should show skeleton rows instead of the empty state while loading", () => {
    renderWithProviders(<RoutingGroupsTable {...defaultProps} isLoading />);
    expect(screen.queryByText("No routing groups yet")).not.toBeInTheDocument();
  });

  describe("URL state", () => {
    it("sorts by the column and direction in the URL", () => {
      renderWithProviders(<RoutingGroupsTable {...defaultProps} groups={[devGroup, prodGroup]} />, {
        searchParams: "?sort_by=routing_strategy&sort_order=desc",
      });
      expect(namesInOrder()).toEqual(["prod-group", "dev-group"]);
    });

    it("ignores a sort_by that is not a sortable column instead of handing it to the table", () => {
      const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});
      try {
        renderWithProviders(<RoutingGroupsTable {...defaultProps} groups={[prodGroup, devGroup]} />, {
          searchParams: "?sort_by=bogus&sort_order=desc",
        });
        expect(namesInOrder()).toEqual(["prod-group", "dev-group"]);
        expect(consoleError).not.toHaveBeenCalledWith(expect.stringContaining("Column with id 'bogus' does not exist"));
      } finally {
        consoleError.mockRestore();
      }
    });

    it("writes header sort clicks to sort_by and sort_order", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<RoutingGroupsTable {...defaultProps} groups={[prodGroup, devGroup]} />, { onUrlUpdate });

      await user.click(screen.getByTestId("sort-header-group_name"));
      expect(lastUrl(onUrlUpdate)?.get("sort_by")).toBe("group_name");
      expect(lastUrl(onUrlUpdate)?.has("sort_order")).toBe(false);

      await user.click(screen.getByTestId("sort-header-group_name"));
      expect(lastUrl(onUrlUpdate)?.get("sort_by")).toBe("group_name");
      expect(lastUrl(onUrlUpdate)?.get("sort_order")).toBe("desc");
      expect(namesInOrder()).toEqual(["prod-group", "dev-group"]);
    });

    it("shows the page and page size from the URL", () => {
      renderWithProviders(<RoutingGroupsTable {...defaultProps} groups={manyGroups} />, {
        searchParams: "?page=2",
      });
      expect(namesInOrder()).toEqual(groupNames.slice(25));
      expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 2 of 2");
    });

    it("shows every row when page_size in the URL covers them", () => {
      renderWithProviders(<RoutingGroupsTable {...defaultProps} groups={manyGroups} />, {
        searchParams: "?page_size=50",
      });
      expect(namesInOrder()).toEqual(groupNames);
      expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 1");
    });

    it("opens an expanded group that sits on the page named in the URL", async () => {
      renderWithProviders(<RoutingGroupsTable {...defaultProps} groups={manyGroups} />, {
        searchParams: "?page=2&expanded=group-27",
      });
      expect(await screen.findAllByText("How routing works for this group")).toHaveLength(1);
      expect(namesInOrder()).toContain("group-27");
    });

    it("writes page changes to page and drops it on the first page", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<RoutingGroupsTable {...defaultProps} groups={manyGroups} />, { onUrlUpdate });

      await user.click(screen.getByRole("button", { name: "Go to next page" }));
      expect(lastUrl(onUrlUpdate)?.get("page")).toBe("2");
      expect(namesInOrder()).toEqual(groupNames.slice(25));

      await user.click(screen.getByRole("button", { name: "Go to previous page" }));
      expect(lastUrl(onUrlUpdate)?.has("page")).toBe(false);
      expect(namesInOrder()).toEqual(groupNames.slice(0, 25));
    });

    it("returns to the first page when the sort changes", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<RoutingGroupsTable {...defaultProps} groups={manyGroups} />, {
        searchParams: "?page=2",
        onUrlUpdate,
      });

      await user.click(screen.getByTestId("sort-header-group_name"));
      expect(lastUrl(onUrlUpdate)?.get("sort_by")).toBe("group_name");
      expect(lastUrl(onUrlUpdate)?.has("page")).toBe(false);
      expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 2");
    });

    it("opens the usage panel for every group named in expanded", async () => {
      renderWithProviders(<RoutingGroupsTable {...defaultProps} groups={[prodGroup, devGroup]} />, {
        searchParams: "?expanded=dev-group",
      });
      expect(await screen.findAllByText("How routing works for this group")).toHaveLength(1);
      const rowSequence = screen
        .getAllByRole("row")
        .slice(1)
        .map(
          (row) =>
            row.getAttribute("data-row-id") ??
            (within(row).queryByText("How routing works for this group") ? "usage-panel" : "other"),
        );
      expect(rowSequence).toEqual(["prod-group", "dev-group", "usage-panel"]);
    });

    it("adds and removes group names in expanded as panels are toggled", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<RoutingGroupsTable {...defaultProps} groups={[prodGroup, devGroup]} />, { onUrlUpdate });

      await user.click(screen.getByRole("button", { name: "prod-group" }));
      expect(lastUrl(onUrlUpdate)?.get("expanded")).toBe("prod-group");

      await user.click(screen.getByRole("button", { name: "dev-group" }));
      expect(lastUrl(onUrlUpdate)?.get("expanded")).toBe("prod-group,dev-group");
      expect(await screen.findAllByText("How routing works for this group")).toHaveLength(2);

      await user.click(screen.getByRole("button", { name: "prod-group" }));
      expect(lastUrl(onUrlUpdate)?.get("expanded")).toBe("dev-group");

      await user.click(screen.getByRole("button", { name: "dev-group" }));
      expect(lastUrl(onUrlUpdate)?.has("expanded")).toBe(false);
      expect(screen.queryByText("How routing works for this group")).not.toBeInTheDocument();
    });
  });
});
