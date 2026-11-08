import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

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
  metadata: Record<string, string> | null;
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

function mockFetch(runs: FakeRun[], events: FakeEvent[] = [], messages: FakeMessage[] = []) {
  return vi.fn((url: string) => {
    if (url.includes("/runs?limit")) {
      return Promise.resolve({ ok: true, json: () => Promise.resolve({ runs }) });
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

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("WorkflowRuns (migrated onto shared DataTable)", () => {
  it("renders one DataTable row per fetched run", async () => {
    vi.stubGlobal("fetch", mockFetch(RUNS));
    const { container } = render(<WorkflowRuns accessToken="tok" />);

    expect(await screen.findByText("First run")).toBeInTheDocument();
    expect(container.querySelectorAll("tr[data-row-id]")).toHaveLength(2);
  });

  it("opens the detail drawer for the clicked run by firing its detail fetch", async () => {
    const user = userEvent.setup();
    const fetchSpy = mockFetch(RUNS);
    vi.stubGlobal("fetch", fetchSpy);
    render(<WorkflowRuns accessToken="tok" />);

    await user.click(await screen.findByText("First run"));

    await waitFor(() =>
      expect(fetchSpy).toHaveBeenCalledWith(expect.stringContaining("run-aaaaaaaa-1111/events"), expect.anything()),
    );
  });

  it("shows the empty state when there are no runs", async () => {
    vi.stubGlobal("fetch", mockFetch([]));
    render(<WorkflowRuns accessToken="tok" />);

    expect(await screen.findByText("No workflow runs yet")).toBeInTheDocument();
  });

  it("sends the configured litellm key header on every fetch instead of hardcoding Authorization", async () => {
    const user = userEvent.setup();
    const fetchSpy = mockFetch(RUNS);
    vi.stubGlobal("fetch", fetchSpy);
    render(<WorkflowRuns accessToken="tok" />);

    await user.click(await screen.findByText("First run"));

    await waitFor(() => expect(fetchSpy).toHaveBeenCalledTimes(3));
    for (const [url, init] of fetchSpy.mock.calls as [string, RequestInit][]) {
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

const DETAIL_RUN: FakeRun = {
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

async function openRunDrawer(run: FakeRun, events: FakeEvent[], messages: FakeMessage[]) {
  const user = userEvent.setup();
  const fetchSpy = mockFetch([run], events, messages);
  vi.stubGlobal("fetch", fetchSpy);
  render(<WorkflowRuns accessToken="tok" />);

  await user.click(await screen.findByText(run.metadata?.title ?? run.workflow_type));
  const drawer = await screen.findByRole("dialog");
  expect(await within(drawer).findByText("Timeline")).toBeInTheDocument();
  return { user, fetchSpy, drawer };
}

const openDetailDrawer = (events = DETAIL_EVENTS, messages = DETAIL_MESSAGES) =>
  openRunDrawer(DETAIL_RUN, events, messages);

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

const FULL_RUN: FakeRun = {
  run_id: "run-cccccccc-3333",
  status: "running",
  workflow_type: "autoship",
  created_at: "2026-01-01T00:00:00.000Z",
  metadata: {
    title: "Full run",
    state: "reviewing",
    pr_url: "https://example.com/pr/2",
    worktree_path: "/tmp/wt/full",
    grill_session_id: "grill-1",
    session_id: "sess-1",
    plan_text: "step ".repeat(26),
  },
};

const TIMED_EVENTS: FakeEvent[] = [
  {
    event_id: "ev-a",
    event_type: "step.started",
    step_name: "plan",
    sequence_number: 1,
    created_at: "2026-01-01T00:00:00.100Z",
    data: null,
  },
  {
    event_id: "ev-b",
    event_type: "step.started",
    step_name: "implement",
    sequence_number: 2,
    created_at: "2026-01-01T00:00:00.165Z",
    data: null,
  },
  {
    event_id: "ev-c",
    event_type: "hook.waiting",
    step_name: "review",
    sequence_number: 3,
    created_at: "2026-01-01T00:00:01.365Z",
    data: null,
  },
];

describe("WorkflowRuns page copy", () => {
  it("announces the loading state while runs are being fetched", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => new Promise(() => {})),
    );
    render(<WorkflowRuns accessToken="tok" />);

    expect(await screen.findByText("Loading workflow runs…")).toBeInTheDocument();
  });

  it("labels the page header, the table columns and the search box", async () => {
    vi.stubGlobal("fetch", mockFetch(RUNS));
    render(<WorkflowRuns accessToken="tok" />);
    await screen.findByText("First run");

    expect(screen.getByText("Workflow Runs")).toBeInTheDocument();
    expect(screen.getByText("Durable state tracking for agents and automated workflows")).toBeInTheDocument();
    expect(screen.getAllByRole("columnheader").map((header) => header.textContent)).toEqual([
      "Run",
      "Type",
      "Status",
      "Created",
    ]);
    expect(screen.getByPlaceholderText("Search runs…")).toBeInTheDocument();
  });

  it("names every column in the Columns menu", async () => {
    const user = userEvent.setup();
    vi.stubGlobal("fetch", mockFetch(RUNS));
    render(<WorkflowRuns accessToken="tok" />);
    await screen.findByText("First run");

    await user.click(screen.getByRole("button", { name: "Columns" }));

    expect((await screen.findAllByRole("menuitemcheckbox")).map((item) => item.textContent)).toEqual([
      "Run",
      "Type",
      "Status",
      "Created",
    ]);
  });

  it("labels the filters drawer and offers every run status as a filter", async () => {
    const user = userEvent.setup();
    vi.stubGlobal("fetch", mockFetch(RUNS));
    render(<WorkflowRuns accessToken="tok" />);
    await screen.findByText("First run");

    await user.click(screen.getByRole("button", { name: "Filters" }));
    const filters = await screen.findByRole("dialog", { name: "Filters" });
    expect(filters).toHaveAccessibleDescription("Narrow down workflow runs");
    expect(within(filters).getByText("Status")).toBeInTheDocument();
    expect(within(filters).getByText("Type")).toBeInTheDocument();
    expect(within(filters).getByPlaceholderText("Filter by type…")).toBeInTheDocument();

    const statusSelect = within(filters).getByRole("combobox");
    expect(statusSelect).toHaveTextContent("All statuses");
    await user.click(statusSelect);
    expect((await screen.findAllByRole("option")).map((option) => option.textContent)).toEqual([
      "All statuses",
      "Pending",
      "Running",
      "Paused",
      "Completed",
      "Failed",
    ]);

    await user.click(screen.getByRole("option", { name: "Running" }));
    expect(statusSelect).toHaveTextContent("Running");
  });

  it("labels the detail drawer, its controls and every primary metadata field", async () => {
    const { user, drawer } = await openRunDrawer(FULL_RUN, [TIMED_EVENTS[0]], []);

    expect(drawer).toHaveAccessibleName("Workflow run details");
    expect(drawer).toHaveAccessibleDescription("Metadata, timeline and messages for the selected workflow run");
    for (const label of ["status", "created", "pr", "state", "worktree", "grill session", "session"]) {
      expect(within(drawer).getByText(label)).toBeInTheDocument();
    }
    expect(within(drawer).getByRole("button", { name: "close" })).toBeInTheDocument();
    expect(within(drawer).getByRole("button", { name: "Refresh" })).toBeInTheDocument();
    expect(within(drawer).getByText("1 event")).toBeInTheDocument();

    await user.click(within(drawer).getByRole("button", { name: /Messages/ }));
    expect(await within(drawer).findByText("No messages")).toBeInTheDocument();
  });

  it("counts zero events in the timeline header", async () => {
    const { drawer } = await openRunDrawer(FULL_RUN, [], []);

    expect(within(drawer).getByText("0 events")).toBeInTheDocument();
  });

  it("collapses a long metadata value behind a more toggle that flips to less", async () => {
    const { user, drawer } = await openRunDrawer(FULL_RUN, [], []);

    await user.click(within(drawer).getByRole("button", { name: "more" }));

    expect(within(drawer).getByRole("button", { name: "less" })).toBeInTheDocument();
  });

  it("formats timeline durations in milliseconds under a second and in seconds above it", async () => {
    const { drawer } = await openRunDrawer(FULL_RUN, TIMED_EVENTS, []);

    expect(within(drawer).getByText("65ms")).toBeInTheDocument();
    expect(within(drawer).getByText("1.2s")).toBeInTheDocument();
    expect(within(drawer).getAllByText("1.4s")).toHaveLength(2);
  });
});

describe("WorkflowRuns relative times", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("shows created times in the largest whole unit and passes an unparseable date through", async () => {
    const now = Date.parse("2026-03-10T12:00:00Z");
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(now);
    const runCreated = (id: string, createdAt: string): FakeRun => ({
      run_id: `run-${id}`,
      status: "completed",
      workflow_type: `type-${id}`,
      created_at: createdAt,
      metadata: null,
    });
    const secondsAgo = (seconds: number) => new Date(now - seconds * 1000).toISOString();
    vi.stubGlobal(
      "fetch",
      mockFetch([
        runCreated("a", secondsAgo(59)),
        runCreated("b", secondsAgo(60)),
        runCreated("c", secondsAgo(3600)),
        runCreated("d", secondsAgo(86400)),
        runCreated("e", "not-a-date"),
      ]),
    );
    render(<WorkflowRuns accessToken="tok" />);

    expect(await screen.findByText("59s ago")).toBeInTheDocument();
    expect(screen.getByText("1m ago")).toBeInTheDocument();
    expect(screen.getByText("1h ago")).toBeInTheDocument();
    expect(screen.getByText("1d ago")).toBeInTheDocument();
    expect(screen.getByText("not-a-date")).toBeInTheDocument();
  });
});

describe("WorkflowRuns detail drawer relative times", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("labels the run's age, each message's age and every field of a hovered timeline event", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(Date.parse("2026-01-01T00:00:10Z"));
    const { user, drawer } = await openDetailDrawer();

    expect(within(drawer).getByText("10s ago")).toBeInTheDocument();

    await user.hover(within(drawer).getByText("step.started"));
    const tooltip = await screen.findByText(
      (_, element) => element?.getAttribute("data-slot") === "tooltip-content",
      {},
      { timeout: 3000 },
    );
    expect(tooltip).toHaveTextContent('type: step.startedstep: planseq: 1time: 9s agodata: {"attempt":1}');

    await user.click(within(drawer).getByRole("button", { name: /Messages/ }));
    expect(await within(drawer).findByText("8s ago")).toBeInTheDocument();
  });
});
