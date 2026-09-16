import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { afterEach, describe, expect, it, vi } from "vitest";

import { chooseSelectOption, renderWithProviders, testQueryClient } from "../../../../tests/test-utils";
import WorkflowRuns from "./WorkflowRuns";

vi.mock("@/components/networking", () => ({
  proxyBaseUrl: "",
  getGlobalLitellmHeaderName: () => "x-litellm-api-key",
}));

interface FakeRun {
  run_id: string;
  status: string;
  workflow_type: string;
  created_at: string;
  metadata: { title?: string; state?: string } | null;
}

const RUNS: FakeRun[] = [
  {
    run_id: "run-aaaaaaaa-1111",
    status: "completed",
    workflow_type: "grill",
    created_at: "2026-01-01T00:00:00Z",
    metadata: { title: "First run", state: "done" },
  },
  {
    run_id: "run-bbbbbbbb-2222",
    status: "running",
    workflow_type: "autofix",
    created_at: "2026-01-02T00:00:00Z",
    metadata: null,
  },
];

function mockFetch(runs: FakeRun[]) {
  return vi.fn((url: string, _init: RequestInit) => {
    if (url.includes("/runs?limit")) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ runs }) });
    }
    if (url.includes("/events")) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ events: [] }) });
    }
    if (url.includes("/messages")) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ messages: [] }) });
    }
    return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) });
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  testQueryClient.clear();
  localStorage.clear();
});

describe("WorkflowRuns (migrated onto shared DataTable)", () => {
  it("renders one DataTable row per fetched run", async () => {
    vi.stubGlobal("fetch", mockFetch(RUNS));
    const { container } = renderWithProviders(<WorkflowRuns accessToken="tok" />);

    expect(await screen.findByText("First run")).toBeInTheDocument();
    expect(container.querySelectorAll("tr[data-row-id]")).toHaveLength(2);
  });

  it("opens the detail drawer for the clicked run by firing its detail fetch", async () => {
    const user = userEvent.setup();
    const fetchSpy = mockFetch(RUNS);
    vi.stubGlobal("fetch", fetchSpy);
    renderWithProviders(<WorkflowRuns accessToken="tok" />);

    await user.click(await screen.findByText("First run"));

    await waitFor(() =>
      expect(fetchSpy).toHaveBeenCalledWith(expect.stringContaining("run-aaaaaaaa-1111/events"), expect.anything()),
    );
  });

  it("shows the empty state when there are no runs", async () => {
    vi.stubGlobal("fetch", mockFetch([]));
    renderWithProviders(<WorkflowRuns accessToken="tok" />);

    expect(await screen.findByText("No workflow runs yet")).toBeInTheDocument();
  });

  it("sends the configured litellm key header on every fetch instead of hardcoding Authorization", async () => {
    const user = userEvent.setup();
    const fetchSpy = mockFetch(RUNS);
    vi.stubGlobal("fetch", fetchSpy);
    renderWithProviders(<WorkflowRuns accessToken="tok" />);

    await user.click(await screen.findByText("First run"));

    await waitFor(() => expect(fetchSpy).toHaveBeenCalledTimes(3));
    for (const [url, init] of fetchSpy.mock.calls) {
      expect(init.headers, url).toEqual({ "x-litellm-api-key": "Bearer tok" });
    }
  });
});

interface FakeEvent {
  event_id: string;
  event_type: string;
  step_name: string;
  sequence_number: number;
  created_at: string;
  data: Record<string, unknown> | null;
}

interface FakeMessage {
  message_id: string;
  role: string;
  content: string;
  sequence_number: number;
  created_at: string;
}

const DETAIL_RUN = {
  run_id: "run-aaaaaaaa-1111",
  status: "completed",
  workflow_type: "grill",
  created_at: "2026-01-01T00:00:00Z",
  metadata: { title: "First run", state: "done", pr_url: "https://example.com/pr/1", worktree_path: "/tmp/wt" },
};

const DETAIL_EVENTS: FakeEvent[] = [
  {
    event_id: "ev-2",
    event_type: "hook.waiting",
    step_name: "review",
    sequence_number: 2,
    created_at: "2026-01-01T00:00:05Z",
    data: null,
  },
  {
    event_id: "ev-1",
    event_type: "step.started",
    step_name: "plan",
    sequence_number: 1,
    created_at: "2026-01-01T00:00:01Z",
    data: { attempt: 1 },
  },
];

const DETAIL_MESSAGES: FakeMessage[] = [
  {
    message_id: "msg-1",
    role: "user",
    content: "kick off the run",
    sequence_number: 1,
    created_at: "2026-01-01T00:00:02Z",
  },
];

function mockDetailFetch(events: FakeEvent[], messages: FakeMessage[]) {
  return vi.fn((url: string) => {
    if (url.includes("/runs?limit")) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ runs: [DETAIL_RUN] }) });
    }
    if (url.includes("/events")) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ events }) });
    }
    if (url.includes("/messages")) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ messages }) });
    }
    return Promise.resolve({ ok: false, status: 404, json: () => Promise.resolve({}) });
  });
}

async function openDetailDrawer(events = DETAIL_EVENTS, messages = DETAIL_MESSAGES) {
  const user = userEvent.setup();
  const fetchSpy = mockDetailFetch(events, messages);
  vi.stubGlobal("fetch", fetchSpy);
  renderWithProviders(<WorkflowRuns accessToken="tok" />);

  await user.click(await screen.findByText("First run"));
  const drawer = await screen.findByRole("dialog");
  await waitFor(() => expect(within(drawer).getByText("Timeline")).toBeInTheDocument());
  return { user, fetchSpy, drawer };
}

describe("WorkflowRuns detail drawer", () => {
  it("shows the run's identity and metadata fields", async () => {
    const { drawer } = await openDetailDrawer();

    expect(within(drawer).getAllByText("First run")).toHaveLength(2);
    expect(within(drawer).getByText("run-aaaa")).toBeInTheDocument();
    expect(within(drawer).getByText("grill")).toBeInTheDocument();
    expect(within(drawer).getByText("completed")).toBeInTheDocument();
    expect(within(drawer).getByText("done")).toBeInTheDocument();
    expect(within(drawer).getByText("/tmp/wt")).toBeInTheDocument();
    expect(within(drawer).getByRole("link", { name: "https://example.com/pr/1" })).toHaveAttribute(
      "href",
      "https://example.com/pr/1",
    );
  });

  it("renders every event in the timeline, ordered by sequence number", async () => {
    const { drawer } = await openDetailDrawer();

    expect(within(drawer).getByText("2 events")).toBeInTheDocument();

    const stepLabels = within(drawer)
      .getAllByText(/^(plan|review)$/)
      .map((el) => el.textContent);
    expect(stepLabels).toEqual(["plan", "review"]);

    expect(within(drawer).getByText("step.started")).toBeInTheDocument();
    expect(within(drawer).getByText("hook.waiting")).toBeInTheDocument();
  });

  it("says no events were recorded when the run has none", async () => {
    const { drawer } = await openDetailDrawer([], DETAIL_MESSAGES);

    expect(within(drawer).getByText("No events recorded")).toBeInTheDocument();
  });

  it("keeps the messages section collapsed until it is opened", async () => {
    const { user, drawer } = await openDetailDrawer();

    expect(within(drawer).queryByText("kick off the run")).not.toBeInTheDocument();

    await user.click(within(drawer).getByRole("button", { name: /Messages/ }));

    expect(await within(drawer).findByText("kick off the run")).toBeInTheDocument();
    expect(within(drawer).getByText("[user]")).toBeInTheDocument();
  });

  it("refetches events and messages when the drawer's refresh button is clicked", async () => {
    const { user, fetchSpy, drawer } = await openDetailDrawer();

    const eventFetches = () => fetchSpy.mock.calls.filter(([url]) => String(url).includes("/events")).length;
    expect(eventFetches()).toBe(1);

    await user.click(within(drawer).getByRole("button", { name: /refresh/i }));

    await waitFor(() => expect(eventFetches()).toBe(2));
  });

  it("dismisses the drawer when its close control is clicked", async () => {
    const { user, drawer } = await openDetailDrawer();

    await user.click(within(drawer).getByRole("button", { name: /close/i }));

    await waitFor(() => expect(screen.queryAllByRole("dialog")).toHaveLength(0));
  });
});

const rowIds = (): string[] =>
  screen
    .queryAllByRole("row")
    .map((row) => row.getAttribute("data-row-id"))
    .filter((id): id is string => id !== null);

const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
  onUrlUpdate.mock.calls.at(-1)?.[0];

const renderRuns = (runs: FakeRun[], searchParams = "") => {
  const fetchSpy = mockFetch(runs);
  vi.stubGlobal("fetch", fetchSpy);
  const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
  renderWithProviders(<WorkflowRuns accessToken="tok" />, { searchParams, onUrlUpdate });
  return { fetchSpy, onUrlUpdate };
};

const detailFetchUrls = (fetchSpy: ReturnType<typeof mockFetch>): string[] =>
  fetchSpy.mock.calls.map(([url]) => String(url)).filter((url) => !url.includes("/runs?limit"));

const STATE_RUNS: FakeRun[] = [
  {
    run_id: "run-labelled",
    status: "completed",
    workflow_type: "grill",
    created_at: "2026-01-01T00:00:00Z",
    metadata: { title: "Labelled run", state: "awaiting_review" },
  },
  {
    run_id: "run-plain",
    status: "completed",
    workflow_type: "autofix",
    created_at: "2026-01-02T00:00:00Z",
    metadata: null,
  },
];

const MANY_RUNS: FakeRun[] = Array.from({ length: 60 }, (_, index) => ({
  run_id: `run-${String(index).padStart(2, "0")}`,
  status: "running",
  workflow_type: "grill",
  created_at: "2026-01-01T00:00:00Z",
  metadata: { title: `Run number ${index}` },
}));

describe("WorkflowRuns status filter", () => {
  it("matches the state the status cell renders rather than the raw run status", async () => {
    renderRuns(STATE_RUNS, "?filter_status=awaiting_review");

    await waitFor(() => expect(rowIds()).toEqual(["run-labelled"]));
    expect(screen.getByTestId("filter-chip-status")).toHaveTextContent("awaiting_review");
  });

  it("leaves out a run whose rendered state differs from the selected raw status", async () => {
    renderRuns(STATE_RUNS, "?filter_status=completed");

    await waitFor(() => expect(rowIds()).toEqual(["run-plain"]));
  });

  it("offers the rendered states in the drawer and writes the chosen one to the URL", async () => {
    const user = userEvent.setup();
    const { onUrlUpdate } = renderRuns(STATE_RUNS);
    await screen.findByText("Labelled run");

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    await chooseSelectOption(user, await screen.findByTestId("filter-status"), "awaiting_review");
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("filter_status")).toBe("awaiting_review"));
    expect(rowIds()).toEqual(["run-labelled"]);
  });
});

describe("WorkflowRuns URL table state", () => {
  it("applies the search from the URL", async () => {
    renderRuns(RUNS, "?search=autofix");

    await waitFor(() => expect(rowIds()).toEqual(["run-bbbbbbbb-2222"]));
    expect(screen.getByTestId("datatable-search")).toHaveValue("autofix");
  });

  it("writes the search to the URL as it is typed", async () => {
    const { onUrlUpdate } = renderRuns(RUNS);
    await screen.findByText("First run");

    fireEvent.change(screen.getByTestId("datatable-search"), { target: { value: "grill" } });

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("search")).toBe("grill"));
    expect(rowIds()).toEqual(["run-aaaaaaaa-1111"]);
  });

  it("applies the type filter from ?filter_type", async () => {
    renderRuns(RUNS, "?filter_type=auto");

    await waitFor(() => expect(rowIds()).toEqual(["run-bbbbbbbb-2222"]));
    expect(screen.getByTestId("filter-chip-workflow_type")).toHaveTextContent("auto");
  });

  it("writes the type filter applied in the drawer to ?filter_type", async () => {
    const user = userEvent.setup();
    const { onUrlUpdate } = renderRuns(RUNS);
    await screen.findByText("First run");

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    await user.type(await screen.findByPlaceholderText("Filter by type…"), "grill");
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("filter_type")).toBe("grill"));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("filter_workflow_type")).toBe(false);
    expect(rowIds()).toEqual(["run-aaaaaaaa-1111"]);
  });

  it("opens the page named in the URL", async () => {
    renderRuns(MANY_RUNS, "?page=2");

    await waitFor(() => expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 2 of 2"));
    expect(rowIds()).toHaveLength(10);
    expect(rowIds()[0]).toBe("run-50");
  });

  it("uses the page size from the URL", async () => {
    renderRuns(MANY_RUNS, "?page_size=100");

    await waitFor(() => expect(rowIds()).toHaveLength(60));
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 1");
  });

  it("writes the page to the URL when paging forward", async () => {
    const user = userEvent.setup();
    const { onUrlUpdate } = renderRuns(MANY_RUNS);
    await screen.findByText("Run number 0");

    await user.click(screen.getByRole("button", { name: "Go to next page" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("page")).toBe("2"));
    expect(rowIds()[0]).toBe("run-50");
  });

  it("writes the page size chosen in the pager to the URL", async () => {
    const user = userEvent.setup();
    const { onUrlUpdate } = renderRuns(MANY_RUNS);
    await screen.findByText("Run number 0");

    await chooseSelectOption(user, screen.getByTestId("pagination-page-size"), "100");

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("page_size")).toBe("100"));
    expect(rowIds()).toHaveLength(60);
  });

  it("drops the page from the URL when the search changes", async () => {
    const { onUrlUpdate } = renderRuns(MANY_RUNS, "?page=2");
    await waitFor(() => expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 2 of 2"));

    fireEvent.change(screen.getByTestId("datatable-search"), { target: { value: "Run number" } });

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("search")).toBe("Run number"));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("page")).toBe(false);
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 2");
  });

  it("drops the page from the URL when a drawer filter is applied", async () => {
    const user = userEvent.setup();
    const { onUrlUpdate } = renderRuns(MANY_RUNS, "?page=2");
    await waitFor(() => expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 2 of 2"));

    await user.click(screen.getByTestId("datatable-filters-trigger"));
    await user.type(await screen.findByPlaceholderText("Filter by type…"), "grill");
    await user.click(screen.getByTestId("filter-drawer-apply"));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("filter_type")).toBe("grill"));
    expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("page")).toBe(false);
    expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 1 of 2");
  });
});

describe("WorkflowRuns ?run= drawer", () => {
  it("opens the drawer for the run named in the URL and loads its detail", async () => {
    const { fetchSpy } = renderRuns(RUNS, "?run=run-bbbbbbbb-2222");

    const drawer = await screen.findByRole("dialog");
    await waitFor(() => expect(within(drawer).getByText("Timeline")).toBeInTheDocument());
    expect(within(drawer).getByText("run-bbbb")).toBeInTheDocument();
    expect(detailFetchUrls(fetchSpy)).toEqual([
      "/v1/workflows/runs/run-bbbbbbbb-2222/events",
      "/v1/workflows/runs/run-bbbbbbbb-2222/messages",
    ]);
  });

  it("pushes the clicked run id to the URL", async () => {
    const user = userEvent.setup();
    const { onUrlUpdate } = renderRuns(RUNS);

    await user.click(await screen.findByText("First run"));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("run")).toBe("run-aaaaaaaa-1111"));
    expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("push");
  });

  it("removes the run from the URL when the drawer is closed", async () => {
    const user = userEvent.setup();
    const { onUrlUpdate } = renderRuns(RUNS, "?run=run-aaaaaaaa-1111");
    const drawer = await screen.findByRole("dialog");

    await user.click(await within(drawer).findByRole("button", { name: /close/i }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("run")).toBe(false));
    await waitFor(() => expect(screen.queryAllByRole("dialog")).toHaveLength(0));
  });

  it("says the run was not found when the URL names a run that is not listed", async () => {
    const { fetchSpy } = renderRuns(RUNS, "?run=run-missing");

    const drawer = await screen.findByRole("dialog");
    expect(await within(drawer).findByText("Workflow run not found.")).toBeInTheDocument();
    expect(detailFetchUrls(fetchSpy)).toEqual([]);
  });

  it("keeps the drawer closed when the run param is empty", async () => {
    const { fetchSpy } = renderRuns(RUNS, "?run=");

    await screen.findByText("First run");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(detailFetchUrls(fetchSpy)).toEqual([]);
  });

  it("shows a spinner rather than not-found while the runs list is still loading", async () => {
    const listGate = Promise.withResolvers<void>();
    const fallback = mockFetch(RUNS);
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string, init: RequestInit) => {
        if (url.includes("/runs?limit")) await listGate.promise;
        return fallback(url, init);
      }),
    );
    renderWithProviders(<WorkflowRuns accessToken="tok" />, { searchParams: "?run=run-bbbbbbbb-2222" });

    const drawer = await screen.findByRole("dialog");
    expect(within(drawer).queryByText("Workflow run not found.")).not.toBeInTheDocument();
    expect(within(drawer).queryByText("Timeline")).not.toBeInTheDocument();

    listGate.resolve();

    await waitFor(() => expect(within(drawer).getByText("Timeline")).toBeInTheDocument());
    expect(within(drawer).queryByText("Workflow run not found.")).not.toBeInTheDocument();
  });

  it("removes the run from the URL when the drawer is dismissed with Escape", async () => {
    const user = userEvent.setup();
    const { onUrlUpdate } = renderRuns(RUNS, "?run=run-aaaaaaaa-1111");
    const drawer = await screen.findByRole("dialog");
    await waitFor(() => expect(within(drawer).getByText("Timeline")).toBeInTheDocument());

    await user.keyboard("{Escape}");

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("run")).toBe(false));
    await waitFor(() => expect(screen.queryAllByRole("dialog")).toHaveLength(0));
  });

  it("encodes the run id in the detail request paths", async () => {
    const encodedRun: FakeRun = { ...RUNS[0], run_id: "run/a b", metadata: { title: "Slashed run" } };
    const { fetchSpy } = renderRuns([encodedRun], "?run=run%2Fa%20b");

    const drawer = await screen.findByRole("dialog");
    await waitFor(() => expect(within(drawer).getByText("Timeline")).toBeInTheDocument());
    expect(detailFetchUrls(fetchSpy)).toEqual([
      "/v1/workflows/runs/run%2Fa%20b/events",
      "/v1/workflows/runs/run%2Fa%20b/messages",
    ]);
  });
});

describe("WorkflowRuns column visibility", () => {
  it("hides the columns saved as hidden for this table", async () => {
    localStorage.setItem("litellm_table_columns_workflow-runs", JSON.stringify({ workflow_type: false }));
    renderRuns(RUNS);

    await screen.findByText("First run");
    expect(screen.queryByRole("columnheader", { name: "Type" })).not.toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Status" })).toBeInTheDocument();
  });

  it("saves a column hidden from the Columns picker", async () => {
    const user = userEvent.setup();
    renderRuns(RUNS);
    await screen.findByText("First run");

    await user.click(screen.getByTestId("view-options-trigger"));
    await user.click(await screen.findByTestId("view-option-workflow_type"));

    await waitFor(() => expect(screen.queryByRole("columnheader", { name: "Type" })).not.toBeInTheDocument());
    expect(JSON.parse(localStorage.getItem("litellm_table_columns_workflow-runs") ?? "{}")).toEqual({
      workflow_type: false,
    });
  });
});
