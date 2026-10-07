import { fireEvent, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { testQueryClient } from "@/../tests/test-utils";
import { readRequest, renderWithLens, stubGateway, type GatewayRequest } from "@/../tests/lens-test-utils";
import { DatasetsView } from "./DatasetsView";
import type { Dataset, DatasetCase, DatasetSummary } from "./types";

vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  proxyBaseUrl: "",
  getProxyBaseUrl: () => "",
}));

const refundCase: DatasetCase = {
  id: "case-refund",
  messages: [
    { role: "user", content: "Can I get a refund for order 42?", name: "", tool_calls: [] },
    { role: "assistant", content: "", name: "", tool_calls: [{ name: "lookup_order", arguments: '{"id": 42}' }] },
  ],
  reply: "I issued the refund.",
  tool_calls: [{ name: "issue_refund", arguments: '{"order": 42}' }],
  expected: "",
  included: true,
  source: { trace_id: "trace-1", trace_ref: "", span_id: "span-1", finding_id: "", lens_id: "" },
  agent_version: "",
};
const shippingCase: DatasetCase = {
  ...refundCase,
  id: "case-shipping",
  messages: [{ role: "user", content: "Where is my package?", name: "", tool_calls: [] }],
  reply: "It ships tomorrow.",
  tool_calls: [],
  source: { ...refundCase.source, trace_id: "trace-2", span_id: "" },
};
const revisionOne: Dataset = {
  id: "ds-1",
  name: "Refund agent regressions",
  agent_name: "refund-agent",
  team_id: "",
  created_at: "2026-10-01T10:00:00Z",
  revision: 1,
  created_by: "admin",
  cases: [refundCase],
};
const revisionTwo: Dataset = {
  ...revisionOne,
  revision: 2,
  cases: [{ ...refundCase, expected: "Refunds above $40 need approval" }, shippingCase],
};
const summary: DatasetSummary = {
  id: revisionTwo.id,
  name: revisionTwo.name,
  agent_name: revisionTwo.agent_name,
  revision: revisionTwo.revision,
  case_count: revisionTwo.cases.length,
  updated_at: "2026-10-02T10:00:00Z",
};

let proxy = stubGateway();

const serveDatasets = (path: string, request: GatewayRequest) => {
  if (path === "/lens/datasets") return [summary];
  if (path !== `/lens/datasets/${revisionTwo.id}`) throw new Error(`unexpected GET ${path}`);
  return request.query.revision === "1" ? revisionOne : revisionTwo;
};

const savedBodies = () =>
  proxy.post.mock.calls
    .filter(([path]) => path === `/lens/datasets/${revisionTwo.id}/revisions`)
    .map(([, request]) => request.body);

beforeEach(() => {
  testQueryClient.clear();
  proxy = stubGateway();
  proxy.get.mockImplementation(serveDatasets);
});

describe("DatasetsView", () => {
  it("lists datasets and opens one from the list", async () => {
    const user = userEvent.setup();
    renderWithLens(<DatasetsView />, { searchParams: "?tab=datasets" });

    const row = await screen.findByRole("row", { name: /Refund agent regressions/ });
    expect(
      within(row)
        .getAllByRole("cell")
        .map((cell) => cell.textContent),
    ).toEqual([`${summary.name}refund-agent`, "2", "2", ""]);
    await user.click(row);

    expect(await screen.findByRole("heading", { name: summary.name })).toBeInTheDocument();
    const first = screen.getByRole("row", { name: "Case 1" });
    expect(
      within(first)
        .getAllByRole("cell")
        .map((cell) => cell.textContent),
    ).toEqual(["", "user: Can I get a refund for order 42?I issued the refund.", "2"]);

    await user.click(first);
    const panel = await screen.findByRole("complementary", { name: "Case details" });
    expect(within(panel).getByRole("heading", { name: `Case #case-ref @ ${summary.name}` })).toBeInTheDocument();
    const input = within(panel).getByRole("region", { name: "Input" });
    expect(input).toHaveTextContent("Can I get a refund for order 42?");
    expect(within(input).getByText("lookup_order")).toBeInTheDocument();
    const output = within(panel).getByRole("region", { name: "Output" });
    expect(output).toHaveTextContent("I issued the refund.");
    expect(within(output).getByText("issue_refund")).toBeInTheDocument();
    expect(within(panel).getByRole("textbox", { name: "Expected" })).toHaveValue("Refunds above $40 need approval");
  });

  it("shows an empty state when there are no datasets", async () => {
    proxy.get.mockResolvedValue([]);
    renderWithLens(<DatasetsView />, { searchParams: "?tab=datasets" });

    expect(await screen.findByText("No datasets yet")).toBeInTheDocument();
  });

  it("saves an edited expected answer and an unticked case as the next revision", async () => {
    const user = userEvent.setup();
    proxy.post.mockResolvedValue({ ...revisionTwo, revision: 3 });
    renderWithLens(<DatasetsView />, { searchParams: `?tab=datasets&dataset=${revisionTwo.id}` });

    await user.click(await screen.findByRole("row", { name: "Case 1" }));
    const save = screen.getByRole("button", { name: "Save as revision 3" });
    expect(save).toBeDisabled();
    const panel = await screen.findByRole("complementary", { name: "Case details" });
    fireEvent.change(within(panel).getByRole("textbox", { name: "Expected" }), {
      target: { value: "Refunds need approval" },
    });
    await user.click(screen.getByRole("checkbox", { name: "Include case 2" }));
    expect(screen.getByRole("contentinfo")).toHaveTextContent("2 cases · 1 included");
    await user.click(save);

    await vi.waitFor(() => expect(savedBodies()).toHaveLength(1));
    expect(savedBodies()[0]).toEqual({
      base_revision: 2,
      cases: [
        { ...refundCase, expected: "Refunds need approval" },
        { ...shippingCase, included: false },
      ],
    });
  });

  it("tells the user to reload when someone saved first", async () => {
    const user = userEvent.setup();
    const gateway = vi.mocked(fetch).getMockImplementation() as typeof fetch;
    vi.mocked(fetch).mockImplementation(async (input, init) =>
      (await readRequest(input, init)).method === "POST"
        ? Response.json({ detail: "Dataset changed, reload" }, { status: 409 })
        : gateway(input, init),
    );
    renderWithLens(<DatasetsView />, { searchParams: `?tab=datasets&dataset=${revisionTwo.id}` });

    await user.click(await screen.findByRole("checkbox", { name: "Include case 1" }));
    await user.click(screen.getByRole("button", { name: "Save as revision 3" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Someone saved a newer revision");
    expect(screen.getByRole("button", { name: "Reload" })).toBeInTheDocument();
  });

  it("shows an older revision's content read-only when picked", async () => {
    const user = userEvent.setup();
    renderWithLens(<DatasetsView />, { searchParams: `?tab=datasets&dataset=${revisionTwo.id}` });

    await screen.findByRole("row", { name: "Case 2" });
    await user.selectOptions(screen.getByRole("combobox", { name: "Revision" }), "1");

    expect(
      await screen.findByText("Revision 1 is read-only. Switch to the latest revision to make changes."),
    ).toBeInTheDocument();
    expect(screen.queryByRole("row", { name: "Case 2" })).not.toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "Include case 1" })).toHaveAttribute("aria-disabled", "true");
    expect(screen.queryByRole("button", { name: /Save as revision/ })).not.toBeInTheDocument();

    await user.click(screen.getByRole("row", { name: "Case 1" }));
    const panel = await screen.findByRole("complementary", { name: "Case details" });
    expect(within(panel).getByRole("region", { name: "Expected" })).toHaveTextContent("Not set");
    expect(within(panel).queryByRole("textbox")).not.toBeInTheDocument();
    expect(within(panel).getByRole("switch", { name: "Included" })).toHaveAttribute("aria-disabled", "true");
  });

  it("exports the viewed revision as JSONL with the auth header", async () => {
    const user = userEvent.setup();
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:dataset");
    vi.spyOn(URL, "revokeObjectURL").mockReturnValue(undefined);
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockReturnValue(undefined);
    renderWithLens(<DatasetsView />, {
      accessToken: "sk-admin",
      searchParams: `?tab=datasets&dataset=${revisionTwo.id}&revision=1`,
    });

    await screen.findByText(/Revision 1 is read-only/);
    await user.click(screen.getByRole("button", { name: "Export JSONL" }));

    await vi.waitFor(() =>
      expect(proxy.get).toHaveBeenCalledWith(
        `/lens/datasets/${revisionTwo.id}/export`,
        expect.objectContaining({ query: { revision: "1" }, authorization: "Bearer sk-admin" }),
      ),
    );
  });
});
