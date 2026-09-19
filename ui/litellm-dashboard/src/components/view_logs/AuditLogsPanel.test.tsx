import { QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, type Mock, vi } from "vitest";

import { chooseSelectOption, renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import AuditLogsPanel from "./AuditLogsPanel";
import type { AuditLogEntry } from "./AuditLogsTableColumns";

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

const ROWS: AuditLogEntry[] = [
  {
    id: "log-1",
    updated_at: "2026-07-20T12:00:00Z",
    changed_by: "user-42",
    changed_by_api_key: "sk-hash-abc",
    action: "created",
    table_name: "LiteLLM_TeamTable",
    object_id: "team-obj-123",
    before_value: {},
    updated_values: { foo: "bar" },
  },
  {
    id: "log-2",
    updated_at: "2026-07-20T11:00:00Z",
    changed_by: "user-43",
    changed_by_api_key: "sk-hash-def",
    action: "deleted",
    table_name: "LiteLLM_UserTable",
    object_id: "user-obj-456",
    before_value: { a: 1 },
    updated_values: {},
  },
];

const PAGE_TWO_ROW: AuditLogEntry = {
  ...ROWS[0],
  id: "log-3",
  object_id: "key-obj-789",
  table_name: "LiteLLM_VerificationToken",
};

const auditLogsResponse = (total: number, rows: AuditLogEntry[] = [], page = 1) => ({
  audit_logs: rows,
  total,
  page,
  page_size: PAGE_SIZE,
  total_pages: Math.ceil(total / PAGE_SIZE),
});

const respondWith = (total: number, rows: AuditLogEntry[] = []) =>
  vi.mocked(uiAuditLogsCall).mockResolvedValue(auditLogsResponse(total, rows));

const lastCall = () => vi.mocked(uiAuditLogsCall).mock.calls.at(-1)?.[0];
const sentIdParams = () => ID_PARAM_KEYS.filter((key) => lastCall()?.params?.[key] !== undefined);
const lastUrl = (onUrlUpdate: Mock<OnUrlUpdateFunction>) => onUrlUpdate.mock.calls.at(-1)?.[0];

const defaultProps = {
  accessToken: "sk-test",
  token: "jwt-test",
  userRole: "Admin",
  userID: "user-1",
  isActive: true,
  premiumUser: true,
};

interface UrlOptions {
  searchParams?: Record<string, string>;
  onUrlUpdate?: OnUrlUpdateFunction;
}

const renderPanel = (urlOptions: UrlOptions = {}) =>
  renderWithProviders(<AuditLogsPanel {...defaultProps} />, urlOptions);

interface UrlHarnessProps {
  searchParams: string;
  onUrlUpdate: OnUrlUpdateFunction;
  isActive?: boolean;
}

function UrlHarness({ searchParams, onUrlUpdate, isActive = true }: UrlHarnessProps) {
  return (
    <NuqsTestingAdapter
      searchParams={searchParams}
      onUrlUpdate={onUrlUpdate}
      hasMemory
      resetUrlUpdateQueueOnMount={false}
    >
      <QueryClientProvider client={testQueryClient}>
        <AuditLogsPanel {...defaultProps} isActive={isActive} />
      </QueryClientProvider>
    </NuqsTestingAdapter>
  );
}

const flushUrlWrites = () => new Promise((resolve) => setTimeout(resolve, 150));

const TEXT_FILTERS: { filterId: string; placeholder: string; paramKey: keyof AuditLogsParams; urlKey: string }[] = [
  { filterId: "object_id", placeholder: "Enter object ID…", paramKey: "object_id", urlKey: "audit_filter_object_id" },
  { filterId: "changed_by", placeholder: "Enter user ID…", paramKey: "changed_by", urlKey: "audit_filter_changed_by" },
  { filterId: "team_id", placeholder: "Enter team ID…", paramKey: "object_team_id", urlKey: "audit_filter_team" },
  {
    filterId: "key_hash",
    placeholder: "Enter key hash…",
    paramKey: "object_key_hash",
    urlKey: "audit_filter_key_hash",
  },
];

const SELECT_FILTERS: {
  label: string;
  comboboxIndex: number;
  option: string;
  paramKey: keyof AuditLogsParams;
  value: string;
  urlKey: string;
}[] = [
  {
    label: "Action",
    comboboxIndex: 0,
    option: "Created",
    paramKey: "action",
    value: "created",
    urlKey: "audit_filter_action",
  },
  {
    label: "Table",
    comboboxIndex: 1,
    option: "Teams",
    paramKey: "table_name",
    value: "LiteLLM_TeamTable",
    urlKey: "audit_filter_table",
  },
];

const URL_FILTERS = [
  ...TEXT_FILTERS.map(({ urlKey, paramKey }) => ({ urlKey, paramKey, value: "val-1" })),
  ...SELECT_FILTERS.map(({ urlKey, paramKey, value }) => ({ urlKey, paramKey, value })),
];

describe("AuditLogsPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
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

    fireEvent.change(screen.getByTestId("datatable-search"), { target: { value: "team-abc" } });

    await waitFor(() => expect(lastCall()?.params?.search).toBe("team-abc"));
    expect(lastCall()?.page).toBe(1);
    expect(sentIdParams()).toEqual(["search"]);
  });

  it("trims the search and drops params.search once the box is cleared", async () => {
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderPanel({ searchParams: { audit_search: "  abc" }, onUrlUpdate });
    await waitFor(() => expect(lastCall()?.params?.search).toBe("abc"));

    fireEvent.change(screen.getByTestId("datatable-search"), { target: { value: "" } });

    await waitFor(() => expect(lastCall()?.params?.search).toBeUndefined());
    expect(sentIdParams()).toEqual([]);
    await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
    expect(lastUrl(onUrlUpdate)?.searchParams.has("audit_search")).toBe(false);
  });

  it.each(TEXT_FILTERS)(
    "maps the $filterId drawer filter to params.$paramKey and ?$urlKey=",
    async ({ placeholder, paramKey, urlKey }) => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderPanel({ onUrlUpdate });
      await waitFor(() => expect(uiAuditLogsCall).toHaveBeenCalled());

      await user.click(screen.getByTestId("datatable-filters-trigger"));
      fireEvent.change(await screen.findByPlaceholderText(placeholder), { target: { value: "val-1" } });
      await user.click(screen.getByTestId("filter-drawer-apply"));

      await waitFor(() => expect(lastCall()?.params?.[paramKey]).toBe("val-1"));
      expect(sentIdParams()).toEqual([paramKey]);
      await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get(urlKey)).toBe("val-1"));
    },
  );

  it.each(SELECT_FILTERS)(
    "maps the $label drawer select to params.$paramKey and ?$urlKey=",
    async ({ comboboxIndex, option, paramKey, value, urlKey }) => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderPanel({ onUrlUpdate });
      await waitFor(() => expect(uiAuditLogsCall).toHaveBeenCalled());

      await user.click(screen.getByTestId("datatable-filters-trigger"));
      const triggers = await screen.findAllByRole("combobox");
      await chooseSelectOption(user, triggers[comboboxIndex], option);
      await user.click(screen.getByTestId("filter-drawer-apply"));

      await waitFor(() => expect(lastCall()?.params?.[paramKey]).toBe(value));
      expect(sentIdParams()).toEqual([paramKey]);
      await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get(urlKey)).toBe(value));
    },
  );

  describe("URL state", () => {
    it("queries the page, page size, search and filters named by the audit_ URL keys", async () => {
      respondWith(120);
      renderPanel({
        searchParams: {
          audit_page: "2",
          audit_page_size: "25",
          audit_search: "abc",
          audit_filter_team: "team-1",
          audit_filter_action: "created",
          audit_filter_table: "LiteLLM_TeamTable",
        },
      });

      await waitFor(() => expect(lastCall()?.page).toBe(2));
      expect(lastCall()?.page_size).toBe(25);
      const expectedParams = {
        search: "abc",
        object_team_id: "team-1",
        action: "created",
        table_name: "LiteLLM_TeamTable",
      };
      expect(lastCall()?.params).toMatchObject(expectedParams);
      expect(sentIdParams()).toEqual(["search", "object_team_id", "action", "table_name"]);
      expect(screen.getByTestId("datatable-search")).toHaveValue("abc");
      expect(screen.getByTestId("filter-chip-team_id")).toHaveTextContent("team-1");
      await waitFor(() => expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 2 of 5"));
    });

    it.each(URL_FILTERS)("sends ?$urlKey= as params.$paramKey", async ({ urlKey, paramKey, value }) => {
      renderPanel({ searchParams: { [urlKey]: value } });

      await waitFor(() => expect(lastCall()?.params?.[paramKey]).toBe(value));
      expect(sentIdParams()).toEqual([paramKey]);
    });

    it("writes audit_page_size when the page size changes", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      respondWith(120);
      renderPanel({ onUrlUpdate });
      await waitFor(() => expect(uiAuditLogsCall).toHaveBeenCalled());

      await chooseSelectOption(user, screen.getByTestId("pagination-page-size"), "100");

      await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get("audit_page_size")).toBe("100"));
      expect(lastUrl(onUrlUpdate)?.searchParams.has("page_size")).toBe(false);
      await waitFor(() => expect(lastCall()?.page_size).toBe(100));
    });

    it("keeps ?audit_page= while the tab is inactive, then requests that page once it opens", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      respondWith(120);
      const { rerender } = render(
        <UrlHarness searchParams="?audit_page=3" onUrlUpdate={onUrlUpdate} isActive={false} />,
      );

      await flushUrlWrites();
      expect(onUrlUpdate).not.toHaveBeenCalled();
      expect(uiAuditLogsCall).not.toHaveBeenCalled();

      rerender(<UrlHarness searchParams="?audit_page=3" onUrlUpdate={onUrlUpdate} />);

      await waitFor(() => expect(lastCall()?.page).toBe(3));
      expect(onUrlUpdate).not.toHaveBeenCalled();
    });

    it("keeps ?audit_page= when the audit log request fails", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      vi.mocked(uiAuditLogsCall).mockRejectedValue(new Error("audit logs unavailable"));
      render(<UrlHarness searchParams="?audit_page=3" onUrlUpdate={onUrlUpdate} />);

      await waitFor(() => expect(uiAuditLogsCall).toHaveBeenCalled());
      await flushUrlWrites();

      expect(onUrlUpdate).not.toHaveBeenCalled();
      expect(lastCall()?.page).toBe(3);
    });

    it("ignores the unprefixed page, search and filter keys that belong to Request Logs", async () => {
      renderPanel({ searchParams: { page: "4", search: "nope", filter_team: "other-team" } });

      await waitFor(() => expect(uiAuditLogsCall).toHaveBeenCalled());
      expect(lastCall()?.page).toBe(1);
      expect(sentIdParams()).toEqual([]);
    });

    it("writes audit_page when paging and clears it again when a filter is applied", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      respondWith(120);
      renderPanel({ onUrlUpdate });
      await waitFor(() => expect(uiAuditLogsCall).toHaveBeenCalled());

      await user.click(screen.getByTestId("pagination-next"));

      await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get("audit_page")).toBe("2"));
      expect(lastUrl(onUrlUpdate)?.searchParams.has("page")).toBe(false);

      await user.click(screen.getByTestId("datatable-filters-trigger"));
      fireEvent.change(await screen.findByPlaceholderText("Enter object ID…"), { target: { value: "obj-9" } });
      await user.click(screen.getByTestId("filter-drawer-apply"));

      await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get("audit_filter_object_id")).toBe("obj-9"));
      expect(lastUrl(onUrlUpdate)?.searchParams.has("audit_page")).toBe(false);
    });

    it("writes the typed search to audit_search", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderPanel({ onUrlUpdate });

      fireEvent.change(await screen.findByTestId("datatable-search"), { target: { value: "team-abc" } });

      await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get("audit_search")).toBe("team-abc"));
      expect(lastUrl(onUrlUpdate)?.searchParams.has("search")).toBe(false);
    });

    it("writes the team and table filters under their renamed audit_ keys", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderPanel({ onUrlUpdate });
      await waitFor(() => expect(uiAuditLogsCall).toHaveBeenCalled());

      await user.click(screen.getByTestId("datatable-filters-trigger"));
      fireEvent.change(await screen.findByPlaceholderText("Enter team ID…"), { target: { value: "team-9" } });
      const [, tableTrigger] = await screen.findAllByRole("combobox");
      await chooseSelectOption(user, tableTrigger, "Teams");
      await user.click(screen.getByTestId("filter-drawer-apply"));

      await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get("audit_filter_team")).toBe("team-9"));
      expect(lastUrl(onUrlUpdate)?.searchParams.get("audit_filter_table")).toBe("LiteLLM_TeamTable");
      expect(lastUrl(onUrlUpdate)?.searchParams.has("audit_filter_team_id")).toBe(false);
      expect(lastUrl(onUrlUpdate)?.searchParams.has("audit_filter_table_name")).toBe(false);
    });
  });

  describe("detail drawer deep link", () => {
    it("opens the drawer for ?audit_log_id= when that row is on the loaded page", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      respondWith(2, ROWS);
      render(<UrlHarness searchParams="?audit_log_id=log-2" onUrlUpdate={onUrlUpdate} />);

      const dialog = await screen.findByRole("dialog");
      expect(within(dialog).getByText("user-obj-456")).toBeInTheDocument();
      expect(within(dialog).queryByText("team-obj-123")).not.toBeInTheDocument();
      await flushUrlWrites();
      expect(onUrlUpdate).not.toHaveBeenCalled();
    });

    it("keeps the drawer closed and drops ?audit_log_id= when that row is not on the loaded page", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      respondWith(2, ROWS);
      render(<UrlHarness searchParams="?audit_log_id=log-missing&audit_page_size=25" onUrlUpdate={onUrlUpdate} />);

      await screen.findByText("team-obj-123");
      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      expect(lastUrl(onUrlUpdate)?.searchParams.has("audit_log_id")).toBe(false);
      expect(lastUrl(onUrlUpdate)?.searchParams.get("audit_page_size")).toBe("25");
      expect(lastUrl(onUrlUpdate)?.options.history).toBe("replace");
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });

    it("drops ?audit_log_id= while the Audit Logs tab is inactive", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      respondWith(2, ROWS);
      render(<UrlHarness searchParams="?audit_log_id=log-1" onUrlUpdate={onUrlUpdate} isActive={false} />);

      await waitFor(() => expect(onUrlUpdate).toHaveBeenCalled());
      expect(lastUrl(onUrlUpdate)?.searchParams.has("audit_log_id")).toBe(false);
      expect(uiAuditLogsCall).not.toHaveBeenCalled();
    });

    it("waits for the requested page instead of dropping ?audit_log_id= against the previous page's rows", async () => {
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      const pageTwo = Promise.withResolvers<Awaited<ReturnType<typeof uiAuditLogsCall>>>();
      vi.mocked(uiAuditLogsCall).mockImplementation(({ page }) =>
        page === 2 ? pageTwo.promise : Promise.resolve(auditLogsResponse(120, ROWS)),
      );
      const { rerender } = render(<UrlHarness searchParams="" onUrlUpdate={onUrlUpdate} />);
      await screen.findByText("team-obj-123");

      rerender(<UrlHarness searchParams="?audit_page=2&audit_log_id=log-3" onUrlUpdate={onUrlUpdate} />);
      await waitFor(() => expect(lastCall()?.page).toBe(2));
      await flushUrlWrites();
      expect(onUrlUpdate).not.toHaveBeenCalled();

      pageTwo.resolve(auditLogsResponse(120, [PAGE_TWO_ROW], 2));

      const dialog = await screen.findByRole("dialog");
      expect(within(dialog).getByText("key-obj-789")).toBeInTheDocument();
      expect(onUrlUpdate).not.toHaveBeenCalled();
    });

    it("pushes ?audit_log_id= when a row is opened and removes it on close", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      respondWith(2, ROWS);
      renderPanel({ onUrlUpdate });

      await user.click(await screen.findByText("team-obj-123"));

      await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.get("audit_log_id")).toBe("log-1"));
      expect(lastUrl(onUrlUpdate)?.options.history).toBe("push");
      const dialog = await screen.findByRole("dialog");

      await user.click(within(dialog).getByRole("button", { name: "Close" }));

      await waitFor(() => expect(lastUrl(onUrlUpdate)?.searchParams.has("audit_log_id")).toBe(false));
      await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    });
  });
});
