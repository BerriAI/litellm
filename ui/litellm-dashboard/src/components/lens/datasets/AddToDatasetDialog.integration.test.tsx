import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, expect, it, vi } from "vitest";

import { renderWithLens, stubGateway } from "@/../tests/lens-test-utils";
import { testQueryClient } from "@/../tests/test-utils";

import { AddToDatasetDialog } from "./AddToDatasetDialog";
import type { Dataset, DatasetCase } from "./types";

const source = { trace_id: "trace-1", trace_ref: "", span_id: "", finding_id: "", lens_id: "" };
const makeCase = (id: string, question: string, reply: string): DatasetCase => ({
  id,
  messages: [{ role: "user", content: question, name: "", tool_calls: [] }],
  reply,
  tool_calls: [],
  expected: "",
  included: true,
  source,
  agent_version: "",
});

const kept = makeCase("kept", "Where is order 1042?", "It ships tomorrow.");
const junk = makeCase("junk", "asdf", "I did not understand.");
const existing = makeCase("existing", "Can I return headphones?", "Yes, within 30 days.");
const dataset: Dataset = {
  id: "ds-1",
  name: "support cases",
  agent_name: "support_agent",
  team_id: "",
  created_at: "2026-10-01T00:00:00Z",
  revision: 3,
  created_by: "admin",
  cases: [existing],
};
const sources = [{ kind: "trace" as const, trace_id: "trace-1", trace_ref: "", span_id: "" }];

let proxy = stubGateway();

beforeEach(() => {
  testQueryClient.clear();
  proxy = stubGateway();
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens/datasets")
      return [
        { id: "ds-1", name: "support cases", agent_name: "support_agent", revision: 3, case_count: 1, updated_at: "" },
      ];
    if (path === "/lens/datasets/ds-1") return dataset;
    throw new Error(`unexpected GET ${path}`);
  });
  proxy.post.mockImplementation(async (path) => {
    if (path === "/lens/datasets/build")
      return { cases: [kept, junk], skipped: [{ source: { ...source, trace_id: "trace-2" }, reason: "duplicate" }] };
    if (path === "/lens/datasets/ds-1/revisions") return { ...dataset, revision: 4 };
    throw new Error(`unexpected POST ${path}`);
  });
});

it("saves the dataset's existing cases plus only the ticked new ones, with their expected text", async () => {
  const user = userEvent.setup();
  const onClose = vi.fn();
  renderWithLens(<AddToDatasetDialog sources={sources} agentName="support_agent" onClose={onClose} />);

  const list = await screen.findByRole("list", { name: "Cases to add" });
  expect(within(list).getAllByRole("listitem")[0]).toHaveTextContent("Where is order 1042?It ships tomorrow.");
  expect(within(screen.getByRole("list", { name: "Skipped" })).getByText("Already in the dataset")).toBeVisible();
  expect(proxy.post.mock.calls.find(([path]) => path === "/lens/datasets/build")?.[1].body).toEqual({
    sources,
    dataset_id: "ds-1",
  });

  await user.click(screen.getByRole("checkbox", { name: "Include Case 2" }));
  expect(screen.getByText("1 of 2 selected")).toBeVisible();
  fireEvent.change(screen.getByRole("textbox", { name: "Expected for Case 1" }), {
    target: { value: "Gives the ship date" },
  });
  await user.click(screen.getByRole("button", { name: "Save 1 case" }));

  await waitFor(() => expect(onClose).toHaveBeenCalled());
  const saved = proxy.post.mock.calls.find(([path]) => path === "/lens/datasets/ds-1/revisions")?.[1].body;
  expect(saved).toEqual({
    base_revision: 3,
    cases: [existing, { ...kept, expected: "Gives the ship date" }],
  });
});

it("tells the user to reload when someone else saved the dataset first", async () => {
  const user = userEvent.setup();
  const onClose = vi.fn();
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = new URL(input instanceof Request ? input.url : String(input), "http://localhost");
      const method = input instanceof Request ? input.method : init?.method ?? "GET";
      if (method === "POST" && url.pathname === "/lens/datasets/ds-1/revisions")
        return Response.json({ detail: "Dataset changed, reload" }, { status: 409 });
      if (method === "POST") return Response.json({ cases: [kept], skipped: [] });
      if (url.pathname === "/lens/datasets/ds-1") return Response.json(dataset);
      return Response.json([
        { id: "ds-1", name: "support cases", agent_name: "support_agent", revision: 3, case_count: 1, updated_at: "" },
      ]);
    }),
  );
  renderWithLens(<AddToDatasetDialog sources={sources} agentName="support_agent" onClose={onClose} />);

  await user.click(await screen.findByRole("button", { name: "Save 1 case" }));

  expect(await screen.findByRole("alert")).toHaveTextContent("Reload to add to the latest version");
  expect(screen.getByRole("button", { name: "Save 1 case" })).toBeDisabled();
  expect(onClose).not.toHaveBeenCalled();
});

it("creates a new dataset by name and saves the cases into its first revision", async () => {
  const user = userEvent.setup();
  const created: Dataset = { ...dataset, id: "ds-new", name: "refund cases", revision: 0, cases: [] };
  proxy.get.mockImplementation(async (path) => {
    if (path === "/lens/datasets") return [];
    throw new Error(`unexpected GET ${path}`);
  });
  proxy.post.mockImplementation(async (path) => {
    if (path === "/lens/datasets/build") return { cases: [kept], skipped: [] };
    if (path === "/lens/datasets") return created;
    if (path === "/lens/datasets/ds-new/revisions") return { ...created, revision: 1 };
    throw new Error(`unexpected POST ${path}`);
  });
  renderWithLens(<AddToDatasetDialog sources={sources} agentName="support_agent" onClose={vi.fn()} />);

  fireEvent.change(await screen.findByRole("textbox", { name: "Name" }), { target: { value: "refund cases" } });
  await user.click(await screen.findByRole("button", { name: "Save 1 case" }));

  await waitFor(() =>
    expect(proxy.post.mock.calls.map(([path, request]) => [path, request.body])).toEqual([
      ["/lens/datasets/build", { sources, dataset_id: "" }],
      ["/lens/datasets", { name: "refund cases", agent_name: "support_agent" }],
      ["/lens/datasets/ds-new/revisions", { base_revision: 0, cases: [kept] }],
    ]),
  );
});
