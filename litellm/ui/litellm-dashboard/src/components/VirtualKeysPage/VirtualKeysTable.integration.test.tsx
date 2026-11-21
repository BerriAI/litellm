import { act, fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { KeyResponse } from "../key_team_helpers/key_list";
import { chooseSelectOption, renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import { VirtualKeysTable } from "./VirtualKeysTable";

vi.unmock("@/app/(dashboard)/hooks/useAuthorized");
vi.mock("next/navigation", () => ({
  usePathname: () => "/ui/api-keys/",
  useRouter: () => ({ push: vi.fn() }),
}));

const keys = ["First key", "Second key", "Keep key"].map((key_alias, index) => ({
  token: `hashed-token-${index}`,
  key_alias,
  user_id: "user-1",
  models: [],
  spend: 0,
  max_budget: 100,
})) satisfies Partial<KeyResponse>[];

const keyPage = (rows: readonly Partial<KeyResponse>[] = keys, total = rows.length, page = 1, size = 50) => {
  const response = { keys: rows, total_count: total, current_page: page, total_pages: Math.ceil(total / size) };
  return Response.json(response);
};

const updatedKeysResponse = (rows = keys.slice(0, 2)) =>
  Response.json({
    total_requested: rows.length,
    successful_updates: rows.map(({ token }) => ({ key: token, key_info: {} })),
    failed_updates: [],
  });

const installApi = (rows: readonly Partial<KeyResponse>[] = keys, role = "proxy_admin") => {
  const list = vi.fn<() => Response | Promise<Response>>(() => keyPage(rows));
  const remove = vi.fn<() => Promise<Response>>(async () => Response.json({ deleted_keys: [] }));
  const update = vi.fn<(request: Request) => Promise<Response>>(async () => updatedKeysResponse());
  const responses: Record<string, unknown> = {
    "/ui/litellm-runtime-config.json": {},
    "/litellm/.well-known/litellm-ui-config": {
      server_root_path: "/",
      proxy_base_url: null,
      gateway_base_url: null,
      admin_ui_disabled: false,
    },
    "/management/v1/users/me": { user_id: "user-1", user_email: "user@example.com", user_role: role },
    "/get/ui_settings": { values: {} },
    "/organization/list": [],
    "/v2/team/list": { teams: [], total: 0, page: 1, page_size: 100, total_pages: 1 },
  };
  const fetch: typeof globalThis.fetch = async (input, init) => {
    const request =
      input instanceof Request ? input : new Request(new URL(String(input), window.location.origin), init);
    const path = new URL(request.url).pathname;
    if (path === "/key/list") return list();
    if (path === "/key/delete") return remove();
    if (path === "/key/bulk_update") return update(request);
    if (path in responses) return Response.json(responses[path]);
    throw new Error(`Unexpected request: ${request.method} ${path}`);
  };
  vi.stubGlobal("fetch", fetch);
  return { list, remove, update, responses };
};

const rowCheckbox = (alias: string) =>
  within(screen.getByRole("row", { name: new RegExp(alias) })).getByRole("checkbox");

const selectFirstTwo = async () => {
  await screen.findByRole("row", { name: /First key/ }, { timeout: 5000 });
  expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
  fireEvent.click(await screen.findByRole("button", { name: "Select keys", exact: true }));
  expect(screen.queryByRole("button", { name: "Actions", exact: true })).not.toBeInTheDocument();
  await waitFor(() => expect(rowCheckbox("First key")).not.toHaveAttribute("aria-disabled", "true"));
  fireEvent.click(rowCheckbox("First key"));
  fireEvent.click(rowCheckbox("Second key"));
};

const openAction = async (user: ReturnType<typeof userEvent.setup>, action: "delete" | "budget") => {
  const option = action === "delete" ? "Delete selected (2)" : "Set budget";
  await chooseSelectOption(user, screen.getByRole("button", { name: "Actions", exact: true }), option, "menuitem");
  const dialog = await screen.findByRole("dialog");
  if (action === "budget") {
    fireEvent.change(within(dialog).getByRole("spinbutton", { name: "Budget per key (USD)" }), {
      target: { value: "25" },
    });
  }
  return dialog;
};

beforeEach(() => {
  testQueryClient.clear();
  localStorage.clear();
  const token = `${btoa("{}")}.${btoa(JSON.stringify({ key: "test-token", premium_user: true }))}.test`;
  document.cookie = `token=${token}; path=/`;
});

afterEach(() => {
  testQueryClient.clear();
  document.cookie = "token=; Max-Age=0; path=/";
  vi.unstubAllGlobals();
});

describe("Virtual Keys bulk actions", () => {
  it("keeps selection unavailable when the initial key request fails", async () => {
    const api = installApi();
    api.list.mockRejectedValue(new TypeError("Key list unavailable"));
    renderWithProviders(<VirtualKeysTable />);
    expect(await screen.findByRole("alert", {}, { timeout: 5000 })).toHaveTextContent("Failed to load");
    expect(await screen.findByRole("button", { name: "Select keys", exact: true })).toBeDisabled();
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
  });

  it("requires exact delete confirmation and clears it on cancel and reopen without losing selection", async () => {
    const api = installApi();
    const user = userEvent.setup();
    renderWithProviders(<VirtualKeysTable />);
    await selectFirstTwo();
    const dialog = await openAction(user, "delete");
    const submit = within(dialog).getByRole("button", { name: "Delete", exact: true });
    expect(submit).toBeDisabled();
    for (const value of ["delete", "Confirm delete", "confirm delete "]) {
      fireEvent.change(within(dialog).getByRole("textbox"), { target: { value } });
      expect(submit).toBeDisabled();
    }
    fireEvent.change(within(dialog).getByRole("textbox"), { target: { value: "confirm delete" } });
    expect(submit).toBeEnabled();
    await user.click(within(dialog).getByRole("button", { name: "Cancel", exact: true }));
    expect(rowCheckbox("First key")).toBeChecked();
    expect(rowCheckbox("Second key")).toBeChecked();
    const reopened = await openAction(user, "delete");
    expect(within(reopened).getByRole("textbox")).toHaveValue("");
    expect(within(reopened).getByRole("button", { name: "Delete", exact: true })).toBeDisabled();
    await user.click(within(reopened).getByRole("button", { name: "Cancel", exact: true }));
    await user.click(screen.getByRole("button", { name: "Cancel selection" }));
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
    await user.click(await screen.findByRole("button", { name: "Select keys", exact: true }));
    expect(screen.queryAllByRole("checkbox", { checked: true })).toHaveLength(0);
    expect(screen.queryByRole("button", { name: "Actions", exact: true })).not.toBeInTheDocument();
    expect(api.remove).not.toHaveBeenCalled();
  });

  it("blocks a confirmed deletion if selected rows disappear on refetch", async () => {
    const api = installApi();
    const user = userEvent.setup();
    renderWithProviders(<VirtualKeysTable />);
    await selectFirstTwo();
    const dialog = await openAction(user, "delete");
    fireEvent.change(within(dialog).getByRole("textbox"), { target: { value: "confirm delete" } });
    const submit = within(dialog).getByRole("button", { name: "Delete", exact: true });
    expect(submit).toBeEnabled();
    api.list.mockReturnValue(keyPage([keys[2]]));
    await act(async () => testQueryClient.invalidateQueries({ queryKey: ["keys"] }));
    await waitFor(() => expect(submit).toBeDisabled());
    expect(api.remove).not.toHaveBeenCalled();
  });

  it.each(["delete", "budget"] as const)(
    "preserves selection after mutation and list errors and locks pending %s retries",
    async (action) => {
      const api = installApi();
      const pending = Promise.withResolvers<Response>();
      const mutation = action === "delete" ? api.remove : api.update;
      mutation.mockRejectedValueOnce(new TypeError("Network connection lost")).mockReturnValueOnce(pending.promise);
      const user = userEvent.setup();
      renderWithProviders(<VirtualKeysTable />);
      await selectFirstTwo();
      const firstCheckbox = rowCheckbox("First key");
      const secondCheckbox = rowCheckbox("Second key");
      const cancelSelection = screen.getByRole("button", { name: "Cancel selection" });
      const dialog = await openAction(user, action);
      if (action === "delete") {
        fireEvent.change(within(dialog).getByRole("textbox"), { target: { value: "confirm delete" } });
      }
      const submit = within(dialog).getByRole("button", {
        name: action === "delete" ? "Delete" : "Apply to 2 keys",
        exact: true,
      });
      api.list.mockRejectedValueOnce(new TypeError("Key list unavailable"));
      await user.click(submit);
      expect(await within(dialog).findByText("Network connection lost")).toBeVisible();
      expect(api.list).toHaveBeenCalledTimes(2);
      expect(firstCheckbox).toBeInTheDocument();
      expect(firstCheckbox).toBeChecked();
      expect(secondCheckbox).toBeInTheDocument();
      expect(secondCheckbox).toBeChecked();
      expect(submit).toBeEnabled();
      expect(mutation).toHaveBeenCalledOnce();

      await user.dblClick(submit);
      expect(submit).toBeDisabled();
      expect(cancelSelection).toBeDisabled();
      const cancel = within(dialog).getByRole("button", { name: "Cancel", exact: true });
      expect(cancel).toBeDisabled();
      await user.click(cancel);
      await user.keyboard("{Escape}");
      expect(dialog).toBeInTheDocument();
      expect(mutation).toHaveBeenCalledTimes(2);
      await act(async () =>
        pending.resolve(action === "delete" ? Response.json({ deleted_keys: [] }) : updatedKeysResponse()),
      );
      await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
      expect(screen.getByText("0 keys selected")).toBeVisible();
      expect(rowCheckbox("First key")).not.toBeChecked();
      expect(screen.queryByRole("button", { name: "Actions", exact: true })).not.toBeInTheDocument();
      expect(mutation).toHaveBeenCalledTimes(2);
    },
  );

  it.each(["off-page", "failed"])("retries only the failed key snapshot after %s refetch", async (refetch) => {
    const api = installApi();
    api.list.mockReturnValue(keyPage(keys.slice(0, 2), 3, 1, 2));
    const response = {
      total_requested: 2,
      successful_updates: [{ key: keys[0].token, key_info: {} }],
      failed_updates: [{ key: keys[1].token, key_info: null, failed_reason: "Budget exceeds team limit" }],
    };
    api.update.mockResolvedValueOnce(Response.json(response)).mockResolvedValueOnce(updatedKeysResponse([keys[1]]));
    const user = userEvent.setup();
    renderWithProviders(<VirtualKeysTable />, { searchParams: "?page_size=2&sort_by=max_budget&sort_order=asc" });
    await selectFirstTwo();
    const firstCheckbox = rowCheckbox("First key");
    const secondCheckbox = rowCheckbox("Second key");
    const dialog = await openAction(user, "budget");
    if (refetch === "failed") {
      api.list.mockRejectedValue(new TypeError("Key list unavailable"));
    } else {
      api.list.mockImplementation(() => keyPage([{ ...keys[0], max_budget: 25 }, keys[2]], 3, 1, 2));
    }
    await user.click(within(dialog).getByRole("button", { name: "Apply to 2 keys" }));

    await waitFor(() => expect(firstCheckbox).not.toBeChecked());
    if (refetch === "failed") expect(secondCheckbox).toBeChecked();
    else expect(secondCheckbox).not.toBeInTheDocument();
    expect(firstCheckbox).toBeInTheDocument();
    expect(within(dialog).getByText("Budget exceeds team limit")).toBeInTheDocument();
    expect(dialog).toHaveTextContent("Second key");
    expect(within(dialog).queryByText("First key")).not.toBeInTheDocument();
    const retry = within(dialog).getByRole("button", { name: /Apply to 1 key/ });
    expect(retry).toBeEnabled();
    await user.click(retry);

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(api.update).toHaveBeenCalledTimes(2);
    const body: { keys: { key: string }[] } = await api.update.mock.calls[1][0].clone().json();
    expect(body.keys.map(({ key }) => key)).toEqual([keys[1].token]);
  });

  it.each(["internal_user", "proxy_admin_viewer", "internal_user_viewer"])(
    "restricts bulk actions for %s",
    async (role) => {
      installApi(keys, role);
      const user = userEvent.setup();
      renderWithProviders(<VirtualKeysTable />);
      await screen.findByRole("row", { name: /First key/ });
      if (role === "internal_user") {
        await selectFirstTwo();
        await user.click(screen.getByRole("button", { name: "Actions", exact: true }));
        expect(await screen.findByRole("menuitem", { name: "Delete selected (2)" })).not.toHaveAttribute(
          "aria-disabled",
          "true",
        );
      } else {
        expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
        expect(screen.queryByRole("button", { name: "Select keys", exact: true })).not.toBeInTheDocument();
      }
      expect(
        screen.queryByRole("menuitem", { name: /Move to team|Assign to user|Set budget/ }),
      ).not.toBeInTheDocument();
    },
  );

  it("excludes deleted rows from select-all and prevents moving selected project keys to a team", async () => {
    installApi([{ ...keys[0], project_id: "project-1" }, keys[1], { ...keys[2], deleted_at: "2026-01-02T00:00:00Z" }]);
    const user = userEvent.setup();
    renderWithProviders(<VirtualKeysTable />);
    await screen.findByRole("row", { name: /First key/ });
    await user.click(await screen.findByRole("button", { name: "Select keys", exact: true }));
    await waitFor(() => expect(rowCheckbox("First key")).not.toHaveAttribute("aria-disabled", "true"));
    fireEvent.click(screen.getByRole("checkbox", { name: "Select all rows" }));
    expect(rowCheckbox("First key")).toBeChecked();
    expect(rowCheckbox("Keep key")).toHaveAttribute("aria-disabled", "true");
    expect(rowCheckbox("Keep key")).not.toBeChecked();
    await user.click(screen.getByRole("button", { name: "Actions", exact: true }));
    expect(await screen.findByRole("menuitem", { name: "Move to team" })).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByRole("menuitem", { name: "Assign to user" })).not.toHaveAttribute("aria-disabled", "true");
    expect(screen.getByRole("menuitem", { name: "Set budget" })).not.toHaveAttribute("aria-disabled", "true");
  });

  it.each(["search", "page", "status", "auth"] as const)("exits selection when the %s scope changes", async (scope) => {
    const api = installApi();
    api.list.mockReturnValue(keyPage(keys, 51));
    const user = userEvent.setup();
    renderWithProviders(<VirtualKeysTable />);
    await selectFirstTwo();
    const pending = Promise.withResolvers<Response>();
    api.list.mockReturnValueOnce(pending.promise);
    if (scope === "search") {
      fireEvent.change(screen.getByTestId("datatable-search"), { target: { value: "Keep" } });
      expect(screen.getByRole("button", { name: "Select keys", exact: true })).toBeDisabled();
    } else if (scope === "page") {
      await user.click(screen.getByRole("button", { name: "Go to next page" }));
    } else if (scope === "auth") {
      api.responses["/management/v1/users/me"] = { user_id: "user-2", user_role: "proxy_admin" };
      await act(async () => testQueryClient.invalidateQueries({ queryKey: ["currentUser"] }));
    } else {
      await user.click(screen.getByRole("button", { name: "Filters" }));
      await chooseSelectOption(user, screen.getByRole("combobox", { name: "Status" }), "Deleted");
      await user.click(screen.getByTestId("filter-drawer-apply"));
    }
    await waitFor(() => expect(screen.queryByRole("button", { name: "Cancel selection" })).not.toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "Actions", exact: true })).not.toBeInTheDocument();
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
    expect(api.update).not.toHaveBeenCalled();
    expect(api.remove).not.toHaveBeenCalled();
    if (scope === "auth") return;
    await waitFor(() => expect(api.list).toHaveBeenCalledTimes(2));
    const nextRow = scope === "status" ? { ...keys[2], deleted_at: "2026-01-02T00:00:00Z" } : keys[2];
    await act(async () => pending.resolve(keyPage([nextRow], scope === "page" ? 51 : 1, scope === "page" ? 2 : 1)));
    await screen.findByRole("row", { name: /Keep key/ });
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
  });
});
