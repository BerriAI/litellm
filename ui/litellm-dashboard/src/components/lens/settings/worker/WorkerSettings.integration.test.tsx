import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useQuery } from "@tanstack/react-query";
import { testQueryClient } from "@/../tests/test-utils";
import { readRequest, renderWithLens, type StubbedRequest } from "@/../tests/lens-test-utils";
import { lensQueries } from "../../data/queries";
import { useLensApi } from "../../data/LensServices";
import type { LensList } from "../../model/types";
import { WorkerSettings } from "./WorkerSettings";

vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  proxyBaseUrl: "https://gateway.example/proxy",
}));

type Route = (request: StubbedRequest) => unknown;
const sent: StubbedRequest[] = [];
const network = vi.fn<typeof fetch>();
function serve(route: Route) {
  network.mockImplementation(async (input, init) => {
    const request = await readRequest(input, init);
    sent.push(request);
    const result = route(request);
    return result instanceof Response ? result : Response.json(result ?? null);
  });
}
const calls = (method: string, path: string) =>
  sent.filter((request) => request.method === method && request.path === path);
const writes = () => sent.filter((request) => request.method !== "GET");

const created = {
  token: "",
  managed: true,
  image: "ghcr.io/berriai/litellm-lens-worker:v1.2.3",
  worker: {
    id: "worker",
    name: "Lens worker",
    last_seen: "1970-01-01T00:00:00Z",
    scope: { all_teams: true, api_key_hash: "", team_id: "" },
    revoked: false,
    analysis_key_id: "b".repeat(64),
  },
};

/** Mirrors the workspace: the worker rows come from the cached `/lens` list, so a mutation must refetch it to update them. */
function WorkerSettingsHost() {
  const api = useLensApi();
  const list = useQuery(lensQueries.list(api));
  return list.data ? <WorkerSettings workers={list.data.workers} /> : null;
}

const listCalls = () => calls("GET", "/lens").length;
const keyRoute: Route = ({ path }) =>
  path === "/key/info"
    ? { info: { models: ["analysis-model"], max_budget: 15, budget_duration: "1mo" } }
    : { keys: [{ token: "b".repeat(64), key_alias: "Analysis" }], total_pages: 1 };

describe("Worker setup", () => {
  beforeEach(() => {
    testQueryClient.clear();
    sent.length = 0;
    network.mockReset();
    vi.stubGlobal("fetch", network);
    serve(keyRoute);
  });
  it("enables the installed Lens service without exposing a worker credential", async () => {
    serve((request) => (request.path === "/lens/workers/register" ? created : keyRoute(request)));
    const user = userEvent.setup();
    const { rerender } = renderWithLens(<WorkerSettings workers={[]} />, { accessToken: "admin" });
    await user.click(screen.getByText("Advanced options"));
    await user.click(screen.getByRole("switch", { name: "Use an existing virtual key" }));
    expect(screen.getByRole("button", { name: "Enable investigations" })).toBeDisabled();
    await user.click(screen.getByRole("combobox", { name: "Charge analysis to" }));
    await user.click(await screen.findByRole("option", { name: "Analysis" }));
    await user.click(screen.getByRole("button", { name: "Enable investigations" }));
    expect(calls("POST", "/lens/workers/register").map(({ body }) => body)).toEqual([
      { name: "Lens worker", analysis_key_id: "b".repeat(64), managed: true },
    ]);
    expect(screen.getByRole("status")).toHaveTextContent("Connecting your Lens service");
    expect(screen.queryByRole("button", { name: "Copy Docker command" })).not.toBeInTheDocument();
    rerender(<WorkerSettings workers={[{ ...created.worker, last_seen: new Date().toISOString() }]} />);
    expect(screen.getByRole("heading", { name: "Worker connected" })).toBeVisible();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Done" }));
    expect(screen.getByRole("heading", { name: created.worker.name })).toBeVisible();
    expect(screen.getByText("Connected")).toBeVisible();
  });
  it("assigns billing to an existing worker without replacing its access token", async () => {
    const user = userEvent.setup();
    const workers = vi.fn((): LensList["workers"] => [{ ...created.worker, analysis_key_id: null }]);
    serve((request) => {
      if (request.method === "PUT") {
        workers.mockReturnValue([created.worker]);
        return created.worker;
      }
      if (request.path === "/lens") return { lenses: [], workers: workers(), tracing_enabled: true };
      return keyRoute(request);
    });
    renderWithLens(<WorkerSettingsHost />, { accessToken: "admin" });
    expect(await screen.findByText("Billing key required")).toBeInTheDocument();
    const listedBefore = listCalls();
    await user.click(screen.getByRole("button", { name: "Edit access" }));
    await user.click(screen.getByRole("combobox", { name: "Charge analysis to" }));
    await user.click(await screen.findByRole("option", { name: "Analysis" }));
    await user.click(screen.getByRole("button", { name: "Save analysis access" }));
    expect(calls("PUT", "/lens/workers/worker/billing-key").map(({ body }) => body)).toEqual([
      { analysis_key_id: "b".repeat(64) },
    ]);
    expect(await screen.findByText(/Not connected/)).toBeVisible();
    expect(screen.queryByText("Billing key required")).not.toBeInTheDocument();
    expect(listCalls()).toBe(listedBefore + 1);
    expect(writes().filter((request) => request.method === "POST")).toEqual([]);
  });
  it("requires revoking the current worker before setting up a replacement", async () => {
    const user = userEvent.setup();
    const workers = vi.fn((): LensList["workers"] => [created.worker]);
    serve(({ method, path }) => {
      if (method === "DELETE") {
        workers.mockReturnValue([{ ...created.worker, revoked: true }]);
        return true;
      }
      return path === "/lens" ? { lenses: [], workers: workers(), tracing_enabled: true } : { data: [] };
    });
    renderWithLens(<WorkerSettingsHost />, { accessToken: "admin" });
    const revoke = await screen.findByRole("button", { name: "Revoke access" });
    expect(screen.queryByRole("button", { name: "Add worker" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Enable investigations" })).not.toBeInTheDocument();
    await user.click(revoke);
    expect(calls("DELETE", "/lens/workers/worker")).toHaveLength(1);
    expect(await screen.findByRole("button", { name: "Enable investigations" })).toBeDisabled();
    expect(screen.getByRole("combobox", { name: "Analysis model" })).toBeVisible();
    expect(listCalls()).toBe(2);
  });
  it("cleans up a newly created key when registration fails before retrying", async () => {
    const user = userEvent.setup();
    const posts = [
      Response.json({ token_id: "limited-key-id", key: "sk-secret-not-displayed" }),
      Response.json({ detail: "Registration unavailable" }, { status: 503 }),
      Response.json({}),
      Response.json({ token_id: "retry-key-id" }),
      Response.json(created),
    ];
    serve(({ method, path }) => {
      if (method === "POST") return posts.shift();
      if (path === "/models") return { data: [{ id: "analysis-model" }] };
      if (path === "/lens") return { workers: [] };
      return { keys: [], total_pages: 0 };
    });
    renderWithLens(<WorkerSettings workers={[]} />, { accessToken: "admin" });
    expect(screen.getByRole("button", { name: "Enable investigations" })).toBeDisabled();
    await user.click(screen.getByRole("combobox", { name: "Analysis model" }));
    await user.click(await screen.findByRole("option", { name: "analysis-model" }));
    await user.clear(screen.getByLabelText("Monthly limit (USD)"));
    await user.type(screen.getByLabelText("Monthly limit (USD)"), "12");
    await user.click(screen.getByRole("button", { name: "Enable investigations" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Registration unavailable");
    expect(writes()[0]).toMatchObject({
      path: "/key/generate",
      body: {
        key_alias: "Lens analysis",
        models: ["analysis-model"],
        max_budget: 12,
        budget_duration: "1mo",
        metadata: { purpose: "lens" },
      },
    });
    await user.click(screen.getByRole("button", { name: "Enable investigations" }));
    expect(await screen.findByRole("status")).toHaveTextContent("Connecting your Lens service");
    expect(calls("POST", "/key/delete").map(({ body }) => body)).toEqual([{ keys: ["limited-key-id"] }]);
    expect(writes().map(({ path }) => path)).toEqual([
      "/key/generate",
      "/lens/workers/register",
      "/key/delete",
      "/key/generate",
      "/lens/workers/register",
    ]);
    expect(writes().at(-1)?.body).toEqual({ name: "Lens worker", analysis_key_id: "retry-key-id", managed: true });
    expect(screen.queryByText("sk-secret-not-displayed")).not.toBeInTheDocument();
  });
});
