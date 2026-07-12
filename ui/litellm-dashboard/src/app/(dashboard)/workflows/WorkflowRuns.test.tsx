import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import WorkflowRuns from "./WorkflowRuns";

vi.mock("@/components/networking", () => ({ proxyBaseUrl: "" }));

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
});
