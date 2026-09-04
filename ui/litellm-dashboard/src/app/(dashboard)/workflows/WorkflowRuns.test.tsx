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
  return vi.fn((url: string) => {
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
  render(<WorkflowRuns accessToken="tok" />);

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
