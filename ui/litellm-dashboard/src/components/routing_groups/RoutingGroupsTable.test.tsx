import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

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

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render every column header", () => {
    render(<RoutingGroupsTable {...defaultProps} />);
    for (const header of ["Group Name", "Models", "Strategy"]) {
      expect(screen.getByText(header)).toBeInTheDocument();
    }
  });

  it("should show the empty state when there are no groups", () => {
    render(<RoutingGroupsTable {...defaultProps} />);
    expect(screen.getByText("No routing groups yet")).toBeInTheDocument();
  });

  it("should render the group name, its models, and a human-readable strategy label", () => {
    render(<RoutingGroupsTable {...defaultProps} groups={[prodGroup]} />);
    const row = rowFor("prod-group");
    expect(within(row).getByText("prod-group")).toBeInTheDocument();
    expect(within(row).getByText("gpt-4o")).toBeInTheDocument();
    expect(within(row).getByText("claude-sonnet-4-5")).toBeInTheDocument();
    expect(within(row).getByText("Usage Based")).toBeInTheDocument();
  });

  it("should fall back to the raw strategy value when it has no friendly label", () => {
    render(<RoutingGroupsTable {...defaultProps} groups={[{ ...prodGroup, routing_strategy: "custom-strategy" }]} />);
    expect(within(rowFor("prod-group")).getByText("custom-strategy")).toBeInTheDocument();
  });

  it("should collapse models beyond the first three behind a +N more badge", () => {
    const wideGroup: RoutingGroup = { ...prodGroup, models: ["a", "b", "c", "d", "e"] };
    render(<RoutingGroupsTable {...defaultProps} groups={[wideGroup]} />);
    const row = rowFor("prod-group");
    expect(within(row).getByText("+2 more")).toBeInTheDocument();
    expect(within(row).queryByText("d")).not.toBeInTheDocument();
  });

  it("should keep the incoming order until a column is sorted", async () => {
    const user = userEvent.setup();
    render(<RoutingGroupsTable {...defaultProps} groups={[prodGroup, devGroup]} />);

    const namesInOrder = () =>
      screen
        .getAllByRole("row")
        .slice(1)
        .map((row) => row.getAttribute("data-row-id"));

    expect(namesInOrder()).toEqual(["prod-group", "dev-group"]);

    await user.click(screen.getByTestId("sort-header-group_name"));
    expect(namesInOrder()).toEqual(["dev-group", "prod-group"]);
  });

  it("should toggle the usage panel when the group name is clicked", async () => {
    const user = userEvent.setup();
    render(<RoutingGroupsTable {...defaultProps} groups={[prodGroup]} />);

    expect(screen.queryByText("How routing works for this group")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "prod-group" }));
    expect(await screen.findByText("How routing works for this group")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "prod-group" }));
    expect(screen.queryByText("How routing works for this group")).not.toBeInTheDocument();
  });

  it("should build the usage snippet from the proxy base url and the group's first model", async () => {
    const user = userEvent.setup();
    render(<RoutingGroupsTable {...defaultProps} groups={[prodGroup]} />);
    await user.click(screen.getByRole("button", { name: "prod-group" }));

    const panel = (await screen.findByText("How routing works for this group")).closest("div")?.parentElement;
    expect(panel?.textContent).toContain("https://proxy.example.com");
    expect(panel?.textContent).toContain("gpt-4o");
  });

  it("should expand only the clicked group", async () => {
    const user = userEvent.setup();
    render(<RoutingGroupsTable {...defaultProps} groups={[prodGroup, devGroup]} />);

    await user.click(screen.getByRole("button", { name: "dev-group" }));
    expect(await screen.findAllByText("How routing works for this group")).toHaveLength(1);
    expect(within(rowFor("prod-group")).queryByText("How routing works for this group")).not.toBeInTheDocument();
  });

  it("should edit a group through the actions menu", async () => {
    const user = userEvent.setup();
    render(<RoutingGroupsTable {...defaultProps} groups={[prodGroup]} />);
    await user.click(screen.getByTestId("routing-group-actions-prod-group"));
    await user.click(await screen.findByTestId("routing-group-action-edit"));
    expect(onEdit).toHaveBeenCalledWith(prodGroup);
  });

  it("should delete a group through the actions menu", async () => {
    const user = userEvent.setup();
    render(<RoutingGroupsTable {...defaultProps} groups={[prodGroup]} />);
    await user.click(screen.getByTestId("routing-group-actions-prod-group"));
    await user.click(await screen.findByTestId("routing-group-action-delete"));
    expect(onDelete).toHaveBeenCalledWith(prodGroup);
  });

  it("should show skeleton rows instead of the empty state while loading", () => {
    render(<RoutingGroupsTable {...defaultProps} isLoading />);
    expect(screen.queryByText("No routing groups yet")).not.toBeInTheDocument();
  });
});
