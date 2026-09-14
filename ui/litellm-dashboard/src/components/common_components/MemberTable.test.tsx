import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import React from "react";
import { describe, expect, it, vi } from "vitest";

import type { Member } from "@/components/networking";

import { renderWithProviders } from "../../../tests/test-utils";
import MemberTable, { MemberTableColumn, memberRoleOptions } from "./MemberTable";

const MEMBERS: Member[] = [
  { user_id: "u-zed", user_email: "zed@example.com", user_alias: "Zed Ortiz", role: "user" },
  { user_id: "u-nameless", user_email: "mystery@example.com", user_alias: null, role: "user" },
  { user_id: "u-amy", user_email: "amy@example.com", user_alias: "amy chen", role: "admin" },
  { user_id: "u-bob", user_email: null, user_alias: "Bob Lee", role: "user" },
];

const BUDGETS: Record<string, number | null> = { "u-zed": 50, "u-nameless": null, "u-amy": 1000, "u-bob": 5 };

const budgetColumn: MemberTableColumn = {
  title: "Budget",
  key: "budget",
  sortValue: (member) => BUDGETS[member.user_id ?? ""] ?? null,
  render: (member) => <span>{BUDGETS[member.user_id ?? ""] ?? "Unlimited"}</span>,
};

const renderTable = (overrides: Partial<React.ComponentProps<typeof MemberTable>> = {}) => {
  const props = {
    members: MEMBERS,
    canEdit: true,
    onEdit: vi.fn(),
    onDelete: vi.fn(),
    extraColumns: [budgetColumn],
    ...overrides,
  };
  renderWithProviders(<MemberTable {...props} />);
  return props;
};

const rowIds = (): (string | null)[] =>
  Array.from(document.querySelectorAll("tbody tr[data-row-id]")).map((row) => row.getAttribute("data-row-id"));

const search = (value: string) => fireEvent.change(screen.getByTestId("datatable-search"), { target: { value } });

describe("MemberTable display", () => {
  it("shows each member's name, falling back to a dash when there is none", () => {
    renderTable();

    expect(within(screen.getByRole("row", { name: /zed@example\.com/ })).getByText("Zed Ortiz")).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: /^name/i })).toBeInTheDocument();
    const cells = within(screen.getByRole("row", { name: /mystery@example\.com/ })).getAllByRole("cell");
    expect(cells[0]).toHaveTextContent("-");
  });

  it("orders members by name with nameless members last by default", () => {
    renderTable();

    expect(rowIds()).toEqual(["u-amy", "u-bob", "u-zed", "u-nameless"]);
  });

  it("reports the full member count", () => {
    renderTable();

    expect(screen.getByText("4 Members")).toBeInTheDocument();
  });
});

describe("MemberTable search", () => {
  it("matches on name case-insensitively", async () => {
    renderTable();

    search("ZED");

    await waitFor(() => expect(rowIds()).toEqual(["u-zed"]));
  });

  it("matches on email", async () => {
    renderTable();

    search("mystery@");

    await waitFor(() => expect(rowIds()).toEqual(["u-nameless"]));
  });

  it("matches on user id", async () => {
    renderTable();

    search("u-bob");

    await waitFor(() => expect(rowIds()).toEqual(["u-bob"]));
  });

  it("still searches names when the first member has no name", async () => {
    renderTable({ members: [MEMBERS[1], MEMBERS[0]] });

    search("ortiz");

    await waitFor(() => expect(rowIds()).toEqual(["u-zed"]));
  });

  it("does not match on role", async () => {
    renderTable();

    search("admin");

    await waitFor(() => expect(rowIds()).toEqual([]));
    expect(screen.getByText("No members match your search or filters")).toBeInTheDocument();
  });
});

describe("MemberTable sorting", () => {
  it("sorts by email with missing emails last in both directions", async () => {
    const user = userEvent.setup();
    renderTable();

    await user.click(screen.getByTestId("sort-header-user_email"));
    expect(rowIds()).toEqual(["u-amy", "u-nameless", "u-zed", "u-bob"]);

    await user.click(screen.getByTestId("sort-header-user_email"));
    expect(rowIds()).toEqual(["u-zed", "u-nameless", "u-amy", "u-bob"]);
  });

  it("sorts by role", async () => {
    const user = userEvent.setup();
    renderTable();

    await user.click(screen.getByTestId("sort-header-role"));
    expect(rowIds()[0]).toBe("u-amy");

    await user.click(screen.getByTestId("sort-header-role"));
    expect(rowIds()[3]).toBe("u-amy");
  });

  it("sorts an extra column numerically by its sort value with blanks last", async () => {
    const user = userEvent.setup();
    renderTable();

    await user.click(screen.getByTestId("sort-header-budget"));
    expect(rowIds()).toEqual(["u-bob", "u-zed", "u-amy", "u-nameless"]);

    await user.click(screen.getByTestId("sort-header-budget"));
    expect(rowIds()).toEqual(["u-amy", "u-zed", "u-bob", "u-nameless"]);
  });

  it("flips name order on the second click", async () => {
    const user = userEvent.setup();
    renderTable();

    await user.click(screen.getByTestId("sort-header-user_alias"));
    expect(rowIds()).toEqual(["u-zed", "u-bob", "u-amy", "u-nameless"]);
  });

  it("leaves extra columns without a sort value unsortable", () => {
    renderTable({
      extraColumns: [{ title: "Rate Limits", key: "rate_limits", render: () => <span>No Limits</span> }],
    });

    expect(screen.getByRole("columnheader", { name: "Rate Limits" })).toBeInTheDocument();
    expect(screen.queryByTestId("sort-header-rate_limits")).not.toBeInTheDocument();
  });
});

describe("MemberTable role filter", () => {
  it("shows only members with the chosen role and clears on reset", async () => {
    const user = userEvent.setup();
    renderTable({ roleColumnTitle: "Team Role" });

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    await user.click(screen.getByTestId("filter-role"));
    await user.click(await screen.findByRole("option", { name: "admin" }));
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => expect(rowIds()).toEqual(["u-amy"]));
    expect(screen.getByTestId("filter-chip-role")).toHaveTextContent("Team Role");

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    await user.click(screen.getByTestId("filter-drawer-reset"));

    await waitFor(() => expect(rowIds()).toHaveLength(4));
  });

  it("offers the roles present in the roster", () => {
    expect(memberRoleOptions(MEMBERS)).toEqual(["admin", "user"]);
    expect(memberRoleOptions([{ user_id: "x", role: "" }])).toEqual([]);
  });
});

describe("MemberTable actions", () => {
  it("passes the clicked member to onEdit and onDelete", async () => {
    const user = userEvent.setup();
    const { onEdit, onDelete } = renderTable();
    const row = screen.getByRole("row", { name: /amy@example\.com/ });

    await user.click(within(row).getByTestId("edit-member"));
    await user.click(within(row).getByTestId("delete-member"));

    expect(onEdit).toHaveBeenCalledWith(MEMBERS[2]);
    expect(onDelete).toHaveBeenCalledWith(MEMBERS[2]);
  });

  it("hides delete for members the caller excludes", () => {
    renderTable({ showDeleteForMember: (member) => member.role !== "admin" });

    expect(screen.getAllByTestId("delete-member")).toHaveLength(3);
    expect(
      within(screen.getByRole("row", { name: /amy@example\.com/ })).queryByTestId("delete-member"),
    ).not.toBeInTheDocument();
  });

  it("shows the empty text when there are no members at all", () => {
    renderTable({ members: [], emptyText: "No members found" });

    expect(screen.getByText("No members found")).toBeInTheDocument();
  });
});
