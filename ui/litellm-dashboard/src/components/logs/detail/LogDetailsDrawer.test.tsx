import { focusManager, QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LogDetailsDrawer, type LogDetailsDrawerProps } from "./LogDetailsDrawer";
import { sessionSpendLogsCall } from "../../networking";
import { LogEntry } from "../types";
import { AutoRouterModelGroupsProvider } from "@/components/shared/table_cells";

vi.mock("../../networking", () => ({
  sessionSpendLogsCall: vi.fn(),
}));

const mockUseLogDetails = vi.fn(
  (_requestId: string | undefined, _startTime: string | undefined, _enabled: boolean) => ({
    data: null,
    isLoading: false,
  }),
);
vi.mock("@/app/(dashboard)/hooks/logDetails/useLogDetails", () => ({
  useLogDetails: (requestId: string | undefined, startTime: string | undefined, enabled: boolean) =>
    mockUseLogDetails(requestId, startTime, enabled),
}));

const mockUseUserLookup = vi.fn(() => ({ data: undefined }));
vi.mock("@/app/(dashboard)/hooks/users/useUsers", () => ({
  useUserLookup: (userId: string | null) => mockUseUserLookup(userId),
}));

vi.mock("./LogDetailContent", () => ({
  LogDetailContent: ({ logEntry, userEmail }: { logEntry: LogEntry; userEmail?: string }) => (
    <>
      <span>user-email:{userEmail ?? "none"}</span>
      <span>showing:{logEntry.request_id}</span>
    </>
  ),
  GuardrailJumpLink: () => null,
}));

vi.mock("./DrawerHeader", () => ({
  DrawerHeader: () => null,
}));

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useAutoRouterModelGroups: vi.fn(() => new Set(["smart-router"])),
}));

// jsdom has no Element.scrollTo; the session sidebar scrolls its selected row into view with it.
beforeEach(() => {
  Object.defineProperty(HTMLElement.prototype, "scrollTo", {
    configurable: true,
    value(this: HTMLElement, options?: ScrollToOptions) {
      if (options?.top !== undefined) this.scrollTop = options.top;
    },
  });
});

afterEach(() => {
  delete (HTMLElement.prototype as Partial<HTMLElement>).scrollTo;
});

const makeLog = (overrides: Partial<LogEntry>): LogEntry => ({
  request_id: "req",
  api_key: "",
  team_id: "",
  model: "",
  model_id: "",
  call_type: "acompletion",
  spend: 0,
  total_tokens: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  startTime: "2026-07-08T10:00:00.000Z",
  endTime: "2026-07-08T10:00:01.000Z",
  cache_hit: "false",
  messages: [],
  response: {},
  ...overrides,
});

const sessionLogs = [
  makeLog({
    request_id: "llm-early",
    model: "llm-early",
    startTime: "2026-07-08T10:00:00.000Z",
    endTime: "2026-07-08T10:00:02.000Z",
  }),
  makeLog({
    request_id: "mcp-early",
    model: "tool-early",
    call_type: "call_mcp_tool",
    startTime: "2026-07-08T10:00:01.000Z",
    endTime: "2026-07-08T10:00:06.000Z",
  }),
  makeLog({
    request_id: "llm-late",
    model: "llm-late",
    startTime: "2026-07-08T10:00:02.000Z",
    endTime: "2026-07-08T10:00:05.000Z",
  }),
  makeLog({
    request_id: "mcp-late",
    model: "tool-late",
    call_type: "call_mcp_tool",
    startTime: "2026-07-08T10:00:03.000Z",
    endTime: "2026-07-08T10:00:03.500Z",
  }),
];

const renderSessionDrawer = () => {
  vi.mocked(sessionSpendLogsCall).mockResolvedValue({ data: sessionLogs, total: 4, total_pages: 1 });
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const drawer = (open: boolean) => (
    <QueryClientProvider client={queryClient}>
      <LogDetailsDrawer open={open} onClose={() => {}} logEntry={null} sessionId="session-1" accessToken="token" />
    </QueryClientProvider>
  );
  const { rerender } = render(drawer(true));
  return { rerender, drawer };
};

const sidebarEventNames = () =>
  screen.queryAllByText(/^(llm-early|llm-late|tool-early|tool-late)$/).map((el) => el.textContent);

describe("LogDetailsDrawer session sidebar sorting", () => {
  it("defaults to duration order, longest call first across LLM and MCP calls", async () => {
    renderSessionDrawer();
    await waitFor(() => expect(sidebarEventNames()).toHaveLength(4));
    expect(sidebarEventNames()).toEqual(["tool-early", "llm-late", "llm-early", "tool-late"]);
  });

  it("switches to chronological order across LLM and MCP calls when Start time is selected", async () => {
    renderSessionDrawer();
    await waitFor(() => expect(sidebarEventNames()).toHaveLength(4));

    fireEvent.click(screen.getByText("Start time"));

    await waitFor(() => expect(sidebarEventNames()).toEqual(["llm-early", "tool-early", "llm-late", "tool-late"]));

    fireEvent.click(screen.getByText("Duration"));

    await waitFor(() => expect(sidebarEventNames()).toEqual(["tool-early", "llm-late", "llm-early", "tool-late"]));
  });

  it("resets the sort mode back to duration when the drawer is closed and reopened", async () => {
    const { rerender, drawer } = renderSessionDrawer();
    await waitFor(() => expect(sidebarEventNames()).toHaveLength(4));

    fireEvent.click(screen.getByText("Start time"));
    await waitFor(() => expect(sidebarEventNames()).toEqual(["llm-early", "tool-early", "llm-late", "tool-late"]));

    rerender(drawer(false));
    rerender(drawer(true));

    await waitFor(() => expect(sidebarEventNames()).toEqual(["tool-early", "llm-late", "llm-early", "tool-late"]));
  });
});

describe("LogDetailsDrawer internal user email", () => {
  const renderSingleLog = (user: string | undefined) => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return render(
      <QueryClientProvider client={queryClient}>
        <LogDetailsDrawer
          open
          onClose={() => {}}
          logEntry={makeLog({ request_id: "single", user })}
          accessToken="token"
        />
      </QueryClientProvider>,
    );
  };

  it("looks up the log's internal user and hands the resolved email to the detail content", () => {
    mockUseUserLookup.mockReturnValue({ data: { user_id: "u-1", user_email: "alice@example.com" } });
    renderSingleLog("u-1");

    expect(mockUseUserLookup).toHaveBeenCalledWith("u-1");
    expect(screen.getByText("user-email:alice@example.com")).toBeInTheDocument();
  });

  it("skips the lookup and passes no email when the log has no internal user", () => {
    mockUseUserLookup.mockReturnValue({ data: undefined });
    renderSingleLog(undefined);

    expect(mockUseUserLookup).toHaveBeenCalledWith(null);
    expect(screen.getByText("user-email:none")).toBeInTheDocument();
  });
});

describe("LogDetailsDrawer session sidebar auto-router icon", () => {
  const routedSessionLogs = [
    makeLog({ request_id: "routed", model: "claude-opus-4-8", model_group: "smart-router" }),
    makeLog({ request_id: "direct", model: "claude-haiku-4-5", model_group: "claude-haiku" }),
  ];

  const renderRoutedSession = () => {
    vi.mocked(sessionSpendLogsCall).mockResolvedValue({ data: routedSessionLogs, total: 2, total_pages: 1 });
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return render(
      <QueryClientProvider client={queryClient}>
        <AutoRouterModelGroupsProvider>
          <LogDetailsDrawer open onClose={() => {}} logEntry={null} sessionId="session-1" accessToken="token" />
        </AutoRouterModelGroupsProvider>
      </QueryClientProvider>,
    );
  };

  const rowFor = (label: string): HTMLElement => {
    const row = Array.from(document.body.querySelectorAll("button")).find((button) =>
      button.textContent?.includes(label),
    );
    if (!row) throw new Error(`no sidebar row for ${label}`);
    return row;
  };

  it("marks the auto-routed entry with the router icon and leaves a direct call on the default icon", async () => {
    renderRoutedSession();

    expect(await screen.findByText("claude-opus-4-8")).toBeInTheDocument();

    const routedRow = rowFor("claude-opus-4-8");
    const directRow = rowFor("claude-haiku-4-5");

    expect(routedRow.querySelector(".lucide-waypoints")).not.toBeNull();
    expect(routedRow.querySelector(".lucide-sparkles")).toBeNull();
    expect(directRow.querySelector(".lucide-sparkles")).not.toBeNull();
    expect(directRow.querySelector(".lucide-waypoints")).toBeNull();
  });
});

describe("LogDetailsDrawer large sessions", () => {
  type SessionPage = { data: LogEntry[]; total: number; total_pages: number };

  const PAGE_SIZE = 100;

  const sessionPage = (page: number, total: number, extra: LogEntry[] = []): SessionPage => ({
    data: [
      ...Array.from({ length: Math.min(PAGE_SIZE, total - (page - 1) * PAGE_SIZE) - extra.length }, (_, i) =>
        makeLog({ request_id: `p${page}-${i}`, model: `p${page}-${i}` }),
      ),
      ...extra,
    ],
    total,
    total_pages: Math.ceil(total / PAGE_SIZE),
  });

  const deferredPages = () => {
    const pages = new Map<number, PromiseWithResolvers<SessionPage>>();
    const pageFor = (page: number) => {
      const existing = pages.get(page);
      if (existing) return existing;
      const created = Promise.withResolvers<SessionPage>();
      pages.set(page, created);
      return created;
    };
    const promiseFor = (page: number) => pageFor(page).promise;
    const resolve = (page: number, value: SessionPage) => act(async () => pageFor(page).resolve(value));
    return { promiseFor, resolve };
  };

  const renderLargeSession = (logEntry: LogEntry | null = null) => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    return render(
      <QueryClientProvider client={queryClient}>
        <LogDetailsDrawer open onClose={() => {}} logEntry={logEntry} sessionId="big-session" accessToken="token" />
      </QueryClientProvider>,
    );
  };

  const sessionRowButtons = () =>
    screen.getAllByRole("button").filter((button) => /^(p\d+-\d+|clicked)/.test(button.textContent ?? ""));

  it("shows the first page while the rest of the session is still loading", async () => {
    const pending = deferredPages();
    vi.mocked(sessionSpendLogsCall).mockImplementation(async (_token, _session, { page = 1 } = {}) =>
      page === 1 ? sessionPage(1, 250) : pending.promiseFor(page),
    );
    renderLargeSession();

    expect(await screen.findByText("Loading 100 of 250...")).toBeInTheDocument();
    expect(screen.getByText("0/100 cached")).toBeInTheDocument();

    await pending.resolve(2, sessionPage(2, 250));
    await pending.resolve(3, sessionPage(3, 250));

    expect(await screen.findByText("0/250 cached")).toBeInTheDocument();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("keeps the clicked log selected while its page loads and after it arrives", async () => {
    const longestCall = {
      request_id: "clicked",
      model: "clicked",
      startTime: "2026-07-08T09:00:00.000Z",
      endTime: "2026-07-08T09:10:00.000Z",
    };
    const clicked = makeLog(longestCall);
    const pending = deferredPages();
    vi.mocked(sessionSpendLogsCall).mockImplementation(async (_token, _session, { page = 1 } = {}) =>
      page === 1 ? sessionPage(1, 200) : pending.promiseFor(page),
    );
    renderLargeSession(clicked);

    expect(screen.getByText("showing:clicked")).toBeInTheDocument();
    expect(await screen.findByText("Loading 100 of 200...")).toBeInTheDocument();
    expect(screen.getByText("showing:clicked")).toBeInTheDocument();

    await pending.resolve(2, sessionPage(2, 200, [clicked]));

    expect(await screen.findByRole("button", { current: true })).toHaveTextContent(/^clicked/);
    expect(screen.getByText("showing:clicked")).toBeInTheDocument();
  });

  it("does not download the session again when the window regains focus", async () => {
    vi.mocked(sessionSpendLogsCall).mockResolvedValue({ data: sessionLogs, total: 4, total_pages: 1 });
    renderLargeSession();
    expect(await screen.findByText("0/4 cached")).toBeInTheDocument();
    const callsBeforeFocus = vi.mocked(sessionSpendLogsCall).mock.calls.length;

    try {
      await act(async () => {
        focusManager.setFocused(false);
        focusManager.setFocused(true);
      });
      expect(vi.mocked(sessionSpendLogsCall).mock.calls.length).toBe(callsBeforeFocus);
    } finally {
      focusManager.setFocused(undefined);
    }
  });

  it("mounts only a window of sidebar rows for a session capped at 5,000 logs", async () => {
    vi.mocked(sessionSpendLogsCall).mockImplementation(async (_token, _session, { page = 1 } = {}) =>
      sessionPage(page, 20_000),
    );
    renderLargeSession();

    expect(await screen.findByText("Showing most recent 5000 of 20000", {}, { timeout: 5_000 })).toBeInTheDocument();
    expect(screen.getByText("0/5000 cached")).toBeInTheDocument();
    expect(sessionRowButtons().length).toBeGreaterThan(0);
    expect(sessionRowButtons().length).toBeLessThan(PAGE_SIZE);
  });

  const renderDrawer = (props: Partial<LogDetailsDrawerProps> = {}) => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const drawer = (overrides: Partial<LogDetailsDrawerProps>) => (
      <QueryClientProvider client={queryClient}>
        <LogDetailsDrawer
          open
          onClose={() => {}}
          logEntry={null}
          sessionId="big-session"
          accessToken="token"
          {...props}
          {...overrides}
        />
      </QueryClientProvider>
    );
    const view = render(drawer({}));
    return { rerenderWith: (overrides: Partial<LogDetailsDrawerProps>) => view.rerender(drawer(overrides)) };
  };

  const pageRequests = () => vi.mocked(sessionSpendLogsCall).mock.calls.map(([, , options]) => options?.page);
  const detailFetches = () =>
    mockUseLogDetails.mock.calls.filter(([, , enabled]) => enabled).map(([requestId]) => requestId);

  describe("opened from a link whose log is still loading", () => {
    beforeEach(() => {
      vi.mocked(sessionSpendLogsCall).mockReset();
      mockUseLogDetails.mockClear();
    });

    it("selects and fetches no other log while the linked log is pending", async () => {
      const pending = deferredPages();
      vi.mocked(sessionSpendLogsCall).mockImplementation(async (_token, _session, { page = 1 } = {}) =>
        page === 1 ? sessionPage(1, 200) : pending.promiseFor(page),
      );
      const view = renderDrawer({ linkedLogId: "p2-5", isLinkedLogPending: true });

      await waitFor(() => expect(pageRequests()).toContain(2));
      expect(screen.queryByText(/^showing:/)).not.toBeInTheDocument();
      expect(detailFetches()).toEqual([]);

      view.rerenderWith({
        linkedLogId: null,
        isLinkedLogPending: false,
        logEntry: makeLog({ request_id: "p2-5", model: "p2-5" }),
      });

      expect(screen.getByText("showing:p2-5")).toBeInTheDocument();
      expect(new Set(detailFetches())).toEqual(new Set(["p2-5"]));
    });

    it("selects the linked log as soon as its row loads, matched by litellm_call_id", async () => {
      const linked = makeLog({ request_id: "chatcmpl-linked", model: "linked", litellm_call_id: "call-linked" });
      vi.mocked(sessionSpendLogsCall).mockResolvedValue(sessionPage(1, 100, [linked]));
      renderDrawer({ linkedLogId: "call-linked", isLinkedLogPending: true });

      expect(await screen.findByText("showing:chatcmpl-linked")).toBeInTheDocument();
      expect(new Set(detailFetches())).toEqual(new Set(["chatcmpl-linked"]));
    });

    it("keeps the loaded linked log selected when the lookup finishes without it", async () => {
      const linked = makeLog({ request_id: "chatcmpl-linked", model: "linked", litellm_call_id: "call-linked" });
      vi.mocked(sessionSpendLogsCall).mockResolvedValue(sessionPage(1, 100, [linked]));
      const view = renderDrawer({ linkedLogId: "call-linked", isLinkedLogPending: true });
      expect(await screen.findByText("showing:chatcmpl-linked")).toBeInTheDocument();

      view.rerenderWith({ linkedLogId: "call-linked", isLinkedLogPending: false });

      expect(screen.getByText("showing:chatcmpl-linked")).toBeInTheDocument();
    });

    it("selects the linked log from the first page when the lookup has already finished without it", async () => {
      const linked = makeLog({ request_id: "chatcmpl-linked", model: "linked", litellm_call_id: "call-linked" });
      vi.mocked(sessionSpendLogsCall).mockResolvedValue(sessionPage(1, 100, [linked]));
      renderDrawer({ linkedLogId: "call-linked", isLinkedLogPending: false });

      expect(await screen.findByText("showing:chatcmpl-linked")).toBeInTheDocument();
      expect(new Set(detailFetches())).toEqual(new Set(["chatcmpl-linked"]));
    });

    it("falls back to the most recent log once the lookup has finished and no row matches", async () => {
      vi.mocked(sessionSpendLogsCall).mockResolvedValue(sessionPage(1, 100));
      renderDrawer({ linkedLogId: "missing", isLinkedLogPending: false });

      expect(await screen.findByText("showing:p1-0")).toBeInTheDocument();
    });
  });

  describe("sidebar scroll position while pages arrive", () => {
    const ROW_HEIGHT = 44;
    const VIEWPORT_HEIGHT = 720;
    const START = Date.parse("2026-07-08T10:00:00.000Z");

    const timedLog = (id: string, seconds: number, startsBefore = 0) => {
      const timing = {
        request_id: id,
        model: id,
        startTime: new Date(START - startsBefore).toISOString(),
        endTime: new Date(START - startsBefore + seconds * 1000).toISOString(),
      };
      return makeLog(timing);
    };

    // Newest first, like the session endpoint.
    const timedPage = (page: number, total: number, seconds: number, extra: LogEntry[] = []): SessionPage => ({
      data: [
        ...Array.from({ length: Math.min(PAGE_SIZE, total - (page - 1) * PAGE_SIZE) - extra.length }, (_, i) =>
          timedLog(`p${page}-${i}`, seconds, ((page - 1) * PAGE_SIZE + i) * 60_000),
        ),
        ...extra,
      ],
      total,
      total_pages: Math.ceil(total / PAGE_SIZE),
    });

    const scroller = () => {
      const element = document.querySelector<HTMLElement>('[data-slot="trace-sidebar"]');
      if (!element) throw new Error("no sidebar scroller");
      return element;
    };

    // jsdom has no layout. The tests set how far below the scroller's top the list starts (content
    // such as the guardrail link sits above it) and how tall a rendered row really is.
    const layout = { listOffset: 0, rowHeight: ROW_HEIGHT };
    const offsetHeight = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "offsetHeight");
    const offsetTop = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "offsetTop");
    beforeEach(() => {
      layout.listOffset = 0;
      layout.rowHeight = ROW_HEIGHT;
      Object.defineProperty(HTMLElement.prototype, "offsetHeight", {
        configurable: true,
        get(this: HTMLElement) {
          return this.dataset.slot === "trace-sidebar-row" ? layout.rowHeight : offsetHeight?.get?.call(this);
        },
      });
      Object.defineProperty(HTMLElement.prototype, "offsetTop", {
        configurable: true,
        get(this: HTMLElement) {
          return this.dataset.slot === "trace-sidebar-list" ? layout.listOffset : 0;
        },
      });
    });
    afterEach(() => {
      if (offsetHeight) Object.defineProperty(HTMLElement.prototype, "offsetHeight", offsetHeight);
      if (offsetTop) Object.defineProperty(HTMLElement.prototype, "offsetTop", offsetTop);
    });

    // The virtualizer skips measuring rows it mounts during a scroll and waits for their ResizeObserver
    // entry, which a browser delivers after layout; the shared mock never reports sidebar rows.
    const SharedResizeObserver = global.ResizeObserver;
    beforeEach(() => {
      global.ResizeObserver = class RowResizeObserver extends SharedResizeObserver {
        private readonly onResize: ResizeObserverCallback;
        constructor(callback: ResizeObserverCallback) {
          super(callback);
          this.onResize = callback;
        }
        observe(target: Element, options?: ResizeObserverOptions) {
          super.observe(target, options);
          if (!(target instanceof HTMLElement) || target.dataset.slot !== "trace-sidebar-row") return;
          const box = { inlineSize: 0, blockSize: layout.rowHeight };
          const entry = {
            target,
            contentRect: new DOMRect(0, 0, 0, layout.rowHeight),
            borderBoxSize: [box],
            contentBoxSize: [box],
            devicePixelContentBoxSize: [box],
          };
          queueMicrotask(() => this.onResize([entry], this));
        }
      };
    });
    afterEach(() => {
      global.ResizeObserver = SharedResizeObserver;
    });

    // Where a browser would draw the row: the list's offset plus the row's translateY, less the scroll.
    const offsetOnScreen = (label: string) => {
      const row = screen.queryByText(label)?.closest<HTMLElement>('[data-slot="trace-sidebar-row"]');
      if (!row) throw new Error(`${label} is not rendered in the sidebar`);
      const translateY = /translateY\((-?[\d.]+)px\)/.exec(row.style.transform);
      if (!translateY) throw new Error(`${label} has no translateY`);
      return layout.listOffset + parseFloat(translateY[1]) - scroller().scrollTop;
    };
    // Browsers report a scroll asynchronously, so the test delivers it once the drawer has settled, and waits
    // out the virtualizer's 150 ms "is scrolling" timer so that timer cannot fire between later steps.
    const deliverScroll = () =>
      act(async () => {
        fireEvent.scroll(scroller());
        await new Promise((resolve) => setTimeout(resolve, 200));
      });

    beforeEach(() => {
      vi.mocked(sessionSpendLogsCall).mockReset();
    });

    it("keeps a clicked row where it is on screen when longer calls sort in above it", async () => {
      const pending = deferredPages();
      vi.mocked(sessionSpendLogsCall).mockImplementation(async (_token, _session, { page = 1 } = {}) =>
        page === 1 ? timedPage(1, 200, 1) : pending.promiseFor(page),
      );
      renderDrawer();

      fireEvent.click(await screen.findByText("p1-4"));
      await deliverScroll();
      expect(offsetOnScreen("p1-4")).toBe(4 * ROW_HEIGHT);

      await pending.resolve(2, timedPage(2, 200, 10));
      await deliverScroll();

      expect(screen.getByRole("button", { current: true })).toHaveTextContent(/^p1-4/);
      expect(offsetOnScreen("p1-4")).toBe(4 * ROW_HEIGHT);
      expect(scroller().scrollTop).toBe(PAGE_SIZE * ROW_HEIGHT);
    });

    it("keeps a clicked row in place when rows are a fraction of a pixel taller than a whole number", async () => {
      layout.rowHeight = 38.28;
      const pending = deferredPages();
      vi.mocked(sessionSpendLogsCall).mockImplementation(async (_token, _session, { page = 1 } = {}) =>
        page === 1 ? timedPage(1, 200, 1) : pending.promiseFor(page),
      );
      renderDrawer();

      fireEvent.click(await screen.findByText("p1-4"));
      await deliverScroll();
      const before = offsetOnScreen("p1-4");

      await pending.resolve(2, timedPage(2, 200, 10));
      await deliverScroll();
      await deliverScroll();

      expect(offsetOnScreen("p1-4")).toBeCloseTo(before, 5);
    });

    it("reveals the selected log in full at the end of the list when its rows are taller than estimated", async () => {
      layout.rowHeight = 53;
      // A browser stops a scroll at the end of the content, which is the virtualizer's total size.
      Object.defineProperty(HTMLElement.prototype, "scrollTo", {
        configurable: true,
        value(this: HTMLElement, options?: ScrollToOptions) {
          const list = this.querySelector<HTMLElement>('[data-slot="trace-sidebar-list"]');
          const end = layout.listOffset + parseFloat(list?.style.height ?? "0") - VIEWPORT_HEIGHT;
          if (options?.top !== undefined) this.scrollTop = Math.max(0, Math.min(options.top, end));
        },
      });
      const selected = timedLog("selected", 0.1);
      vi.mocked(sessionSpendLogsCall).mockResolvedValue(timedPage(1, 100, 1, [selected]));
      renderDrawer({ logEntry: selected });

      expect(await screen.findByText("0/100 cached")).toBeInTheDocument();
      await deliverScroll();

      expect(offsetOnScreen("selected")).toBeGreaterThanOrEqual(0);
      expect(offsetOnScreen("selected")).toBeLessThanOrEqual(VIEWPORT_HEIGHT - 53);
    });

    it("keeps the first row on screen in place after the user has scrolled down", async () => {
      const pending = deferredPages();
      vi.mocked(sessionSpendLogsCall).mockImplementation(async (_token, _session, { page = 1 } = {}) =>
        page === 1 ? timedPage(1, 200, 1) : pending.promiseFor(page),
      );
      renderDrawer();
      await screen.findByText("p1-0");

      scroller().scrollTop = 30 * ROW_HEIGHT;
      await deliverScroll();
      await pending.resolve(2, timedPage(2, 200, 10));
      await deliverScroll();

      expect(offsetOnScreen("p1-30")).toBe(0);
    });

    it("centers the selected log when it sorts far below the first screen", async () => {
      const selected = timedLog("selected", 0.1);
      vi.mocked(sessionSpendLogsCall).mockImplementation(async (_token, _session, { page = 1 } = {}) =>
        timedPage(page, 250, 1, page === 3 ? [selected] : []),
      );
      renderDrawer({ logEntry: selected });

      expect(await screen.findByText("0/250 cached")).toBeInTheDocument();
      await deliverScroll();

      expect(screen.getByRole("button", { current: true })).toHaveTextContent(/^selected/);
      expect(offsetOnScreen("selected")).toBe((VIEWPORT_HEIGHT - ROW_HEIGHT) / 2);
    });

    it("centers the selected log when content sits above the list", async () => {
      layout.listOffset = 32;
      const selected = timedLog("selected", 0.1);
      vi.mocked(sessionSpendLogsCall).mockImplementation(async (_token, _session, { page = 1 } = {}) =>
        timedPage(page, 250, 1, page === 3 ? [selected] : []),
      );
      renderDrawer({ logEntry: selected });

      expect(await screen.findByText("0/250 cached")).toBeInTheDocument();
      await deliverScroll();

      expect(offsetOnScreen("selected")).toBe((VIEWPORT_HEIGHT - ROW_HEIGHT) / 2);
    });

    it("keeps the rows on screen in place when content appears above the list on a new selection", async () => {
      vi.mocked(sessionSpendLogsCall).mockResolvedValue(timedPage(1, 100, 1));
      renderDrawer();
      await screen.findByText("p1-0");
      scroller().scrollTop = 30 * ROW_HEIGHT;
      await deliverScroll();
      const before = offsetOnScreen("p1-35");

      layout.listOffset = 32;
      fireEvent.click(screen.getByText("p1-35"));
      await deliverScroll();

      expect(screen.getByRole("button", { current: true })).toHaveTextContent(/^p1-35/);
      expect(offsetOnScreen("p1-35")).toBe(before);
    });

    it("keeps a clicked row in place when rows taller than the estimate sort in right above it", async () => {
      layout.rowHeight = 53;
      // The first ten calls outlast the next page and the rest do not, so the new rows land between them.
      const firstPage: SessionPage = {
        data: Array.from({ length: PAGE_SIZE }, (_, i) => timedLog(`p1-${i}`, i < 10 ? 5 : 1, i * 60_000)),
        total: 200,
        total_pages: 2,
      };
      const pending = deferredPages();
      vi.mocked(sessionSpendLogsCall).mockImplementation(async (_token, _session, { page = 1 } = {}) =>
        page === 1 ? firstPage : pending.promiseFor(page),
      );
      renderDrawer();

      fireEvent.click(await screen.findByText("p1-10"));
      await deliverScroll();
      expect(offsetOnScreen("p1-10")).toBe(10 * 53);

      await pending.resolve(2, timedPage(2, 200, 3));
      await deliverScroll();
      await deliverScroll();

      expect(screen.getByRole("button", { current: true })).toHaveTextContent(/^p1-10/);
      expect(screen.getByText("p2-99")).toBeInTheDocument();
      expect(offsetOnScreen("p1-10")).toBe(10 * 53);
    });
  });

  describe("session downloads", () => {
    beforeEach(() => {
      vi.mocked(sessionSpendLogsCall).mockReset();
    });

    it("reuses the loaded pages when the same session is opened again", async () => {
      vi.mocked(sessionSpendLogsCall).mockResolvedValue({ data: sessionLogs, total: 4, total_pages: 1 });
      const view = renderDrawer();
      expect(await screen.findByText("0/4 cached")).toBeInTheDocument();
      const requestsBeforeReopen = pageRequests().length;

      view.rerenderWith({ open: false, sessionId: null });
      view.rerenderWith({});

      expect(await screen.findByText("0/4 cached")).toBeInTheDocument();
      await act(async () => {});
      expect(pageRequests()).toHaveLength(requestsBeforeReopen);
    });

    it.each([
      ["the drawer closes", { open: false, sessionId: null }],
      ["another session opens", { sessionId: "other-session" }],
    ])("aborts the pages still downloading when %s", async (_case, change: Partial<LogDetailsDrawerProps>) => {
      const signals: (AbortSignal | undefined)[] = [];
      vi.mocked(sessionSpendLogsCall).mockImplementation(async (_token, session, { page = 1, signal } = {}) => {
        if (page === 1) return sessionPage(1, 300);
        if (session === "big-session") signals.push(signal);
        return new Promise(() => {});
      });
      const view = renderDrawer();
      await waitFor(() => expect(signals).toHaveLength(2));

      view.rerenderWith(change);

      expect(signals.map((signal) => signal?.aborted)).toEqual([true, true]);
    });
  });
});
