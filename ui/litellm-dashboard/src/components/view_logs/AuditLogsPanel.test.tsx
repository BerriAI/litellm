import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { chooseSelectOption } from "../../../tests/test-utils";
import AuditLogsPanel from "./AuditLogsPanel";

vi.mock("../networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../networking")>();
  return { ...actual, uiAuditLogsCall: vi.fn() };
});

// Resolve the debounced search synchronously so typed input reaches the query within the test tick.
vi.mock("@tanstack/react-pacer/debouncer", () => ({
  useDebouncedValue: (value: unknown) => [value, { cancel: vi.fn(), flush: vi.fn() }],
}));

import { uiAuditLogsCall } from "../networking";

type AuditLogsParams = NonNullable<Parameters<typeof uiAuditLogsCall>[0]["params"]>;

const PAGE_SIZE = 50;

const ID_PARAM_KEYS = [
  "search",
  "object_id",
  "changed_by",
  "object_team_id",
  "object_key_hash",
  "action",
  "table_name",
] as const satisfies readonly (keyof AuditLogsParams)[];

const respondWith = (total: number) => {
  const response = { audit_logs: [], total, page: 1, page_size: PAGE_SIZE, total_pages: Math.ceil(total / PAGE_SIZE) };
  return vi.mocked(uiAuditLogsCall).mockResolvedValue(response);
};

const lastCall = () => vi.mocked(uiAuditLogsCall).mock.calls.at(-1)?.[0];
const sentIdParams = () => ID_PARAM_KEYS.filter((key) => lastCall()?.params?.[key] !== undefined);

const defaultProps = {
  accessToken: "sk-test",
  token: "jwt-test",
  userRole: "Admin",
  userID: "user-1",
  isActive: true,
  premiumUser: true,
};

const renderPanel = () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={queryClient}>
      <AuditLogsPanel {...defaultProps} />
    </QueryClientProvider>,
  );
};

const TEXT_FILTERS: { filterId: string; placeholder: string; paramKey: keyof AuditLogsParams }[] = [
  { filterId: "object_id", placeholder: "Enter object ID…", paramKey: "object_id" },
  { filterId: "changed_by", placeholder: "Enter user ID…", paramKey: "changed_by" },
  { filterId: "team_id", placeholder: "Enter team ID…", paramKey: "object_team_id" },
  { filterId: "key_hash", placeholder: "Enter key hash…", paramKey: "object_key_hash" },
];

const SELECT_FILTERS: {
  label: string;
  comboboxIndex: number;
  option: string;
  paramKey: keyof AuditLogsParams;
  value: string;
}[] = [
  { label: "Action", comboboxIndex: 0, option: "Created", paramKey: "action", value: "created" },
  { label: "Table", comboboxIndex: 1, option: "Teams", paramKey: "table_name", value: "LiteLLM_TeamTable" },
];

describe("AuditLogsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    respondWith(0);
  });

  it("sends the typed search as params.search and returns to the first page", async () => {
    const user = userEvent.setup();
    respondWith(120);
    renderPanel();
    await waitFor(() => expect(uiAuditLogsCall).toHaveBeenCalled());
    expect(lastCall()?.params?.search).toBeUndefined();

    await user.click(screen.getByTestId("pagination-next"));
    await waitFor(() => expect(lastCall()?.page).toBe(2));

    await user.type(screen.getByTestId("datatable-search"), "team-abc");

    await waitFor(() => expect(lastCall()?.params?.search).toBe("team-abc"));
    expect(lastCall()?.page).toBe(1);
    expect(sentIdParams()).toEqual(["search"]);
  });

  it("trims the search and drops params.search once the box is cleared", async () => {
    const user = userEvent.setup();
    renderPanel();
    const input = await screen.findByTestId("datatable-search");

    await user.type(input, "  abc");
    await waitFor(() => expect(lastCall()?.params?.search).toBe("abc"));

    await user.clear(input);

    await waitFor(() => expect(lastCall()?.params?.search).toBeUndefined());
    expect(sentIdParams()).toEqual([]);
  });

  it.each(TEXT_FILTERS)("maps the $filterId drawer filter to params.$paramKey", async ({ placeholder, paramKey }) => {
    const user = userEvent.setup();
    renderPanel();
    await waitFor(() => expect(uiAuditLogsCall).toHaveBeenCalled());

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    fireEvent.change(await screen.findByPlaceholderText(placeholder), { target: { value: "val-1" } });
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => expect(lastCall()?.params?.[paramKey]).toBe("val-1"));
    expect(sentIdParams()).toEqual([paramKey]);
  });

  it.each(SELECT_FILTERS)(
    "maps the $label drawer select to params.$paramKey",
    async ({ comboboxIndex, option, paramKey, value }) => {
      const user = userEvent.setup();
      renderPanel();
      await waitFor(() => expect(uiAuditLogsCall).toHaveBeenCalled());

      await user.click(screen.getByTestId("datatable-filters-trigger"));
      const triggers = await screen.findAllByRole("combobox");
      await chooseSelectOption(user, triggers[comboboxIndex], option);
      await user.click(screen.getByTestId("filter-drawer-apply"));

      await waitFor(() => expect(lastCall()?.params?.[paramKey]).toBe(value));
      expect(sentIdParams()).toEqual([paramKey]);
    },
  );
});
