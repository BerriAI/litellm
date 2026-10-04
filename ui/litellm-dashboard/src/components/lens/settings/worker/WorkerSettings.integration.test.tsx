import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useQuery } from "@tanstack/react-query";
import { testQueryClient } from "@/../tests/test-utils";
import { renderWithLens } from "@/../tests/lens-test-utils";
import { apiClient } from "@/components/networking";
import { lensQueries } from "../../data/queries";
import { useLensApi } from "../../data/LensServices";
import type { LensList } from "../../model/types";
import { WorkerSettings } from "./WorkerSettings";

vi.mock("@/components/networking", () => ({
  apiClient: { get: vi.fn(), post: vi.fn(), put: vi.fn(), delete: vi.fn() },
  proxyBaseUrl: "https://gateway.example/proxy",
}));

const created = {
  token: "lens-test-token",
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

const listCalls = () => vi.mocked(apiClient.get).mock.calls.filter(([path]) => path === "/lens").length;

describe("Worker setup", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.clearAllMocks();
    vi.mocked(apiClient.get).mockImplementation(async (path) =>
      path === "/key/info"
        ? { info: { models: ["analysis-model"], max_budget: 15, budget_duration: "1mo" } }
        : {
            keys: [{ token: "b".repeat(64), key_alias: "Analysis" }],
            total_pages: 1,
          },
    );
  });
  it("generates a complete command using one worker credential and the configured proxy address", async () => {
    vi.mocked(apiClient.post).mockResolvedValue(created);
    const user = userEvent.setup();
    const { rerender } = renderWithLens(<WorkerSettings workers={[]} />, { accessToken: "admin" });
    await user.click(screen.getByText("Advanced options"));
    await user.click(screen.getByRole("switch", { name: "Use an existing virtual key" }));
    expect(screen.getByRole("textbox", { name: "LiteLLM proxy URL" })).toHaveValue("https://gateway.example/proxy");
    expect(screen.getByRole("button", { name: "Get install command" })).toBeDisabled();
    await user.click(screen.getByRole("combobox", { name: "Charge analysis to" }));
    await user.click(await screen.findByRole("option", { name: "Analysis" }));
    await user.click(screen.getByRole("button", { name: "Get install command" }));
    expect(apiClient.post).toHaveBeenCalledWith("/lens/workers/register", {
      accessToken: "admin",
      body: { name: "Lens worker", analysis_key_id: "b".repeat(64) },
    });
    expect(screen.getByRole("status")).toHaveTextContent("Waiting for your worker to connect");
    expect(screen.getByLabelText("Docker command preview")).not.toBeVisible();
    await user.click(screen.getByRole("button", { name: "Copy Docker command" }));
    const command = await navigator.clipboard.readText();
    expect(command).toContain("LITELLM_URL=https://gateway.example/proxy");
    expect(command).toContain("LENS_WORKER_TOKEN=lens-test-token");
    expect(command).toContain("--add-host host.docker.internal:host-gateway");
    expect(command).toContain(created.image);
    await user.click(screen.getByText("Using Docker Compose or Helm?"));
    await user.click(screen.getByRole("button", { name: "Copy worker token" }));
    expect(await navigator.clipboard.readText()).toBe(created.token);
    expect(await screen.findByRole("button", { name: "Token copied" })).toBeVisible();
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
    vi.mocked(apiClient.get).mockImplementation(async (path) => {
      if (path === "/lens") return { lenses: [], workers: workers(), tracing_enabled: true };
      if (path === "/key/info") return { info: { models: ["analysis-model"], max_budget: 15, budget_duration: "1mo" } };
      return { keys: [{ token: "b".repeat(64), key_alias: "Analysis" }], total_pages: 1 };
    });
    vi.mocked(apiClient.put).mockImplementation(async () => {
      workers.mockReturnValue([created.worker]);
      return created.worker;
    });
    renderWithLens(<WorkerSettingsHost />, { accessToken: "admin" });
    expect(await screen.findByText("Billing key required")).toBeInTheDocument();
    const listedBefore = listCalls();
    await user.click(screen.getByRole("button", { name: "Edit access" }));
    await user.click(screen.getByRole("combobox", { name: "Charge analysis to" }));
    await user.click(await screen.findByRole("option", { name: "Analysis" }));
    await user.click(screen.getByRole("button", { name: "Save analysis access" }));
    expect(apiClient.put).toHaveBeenCalledWith("/lens/workers/worker/billing-key", {
      accessToken: "admin",
      body: { analysis_key_id: "b".repeat(64) },
    });
    expect(await screen.findByText(/Not connected/)).toBeVisible();
    expect(screen.queryByText("Billing key required")).not.toBeInTheDocument();
    expect(listCalls()).toBe(listedBefore + 1);
    expect(apiClient.post).not.toHaveBeenCalled();
  });
  it("requires revoking the current worker before setting up a replacement", async () => {
    const user = userEvent.setup();
    const workers = vi.fn((): LensList["workers"] => [created.worker]);
    vi.mocked(apiClient.get).mockImplementation(async (path) =>
      path === "/lens" ? { lenses: [], workers: workers(), tracing_enabled: true } : { data: [] },
    );
    vi.mocked(apiClient.delete).mockImplementation(async () => {
      workers.mockReturnValue([{ ...created.worker, revoked: true }]);
      return true;
    });
    renderWithLens(<WorkerSettingsHost />, { accessToken: "admin" });
    const revoke = await screen.findByRole("button", { name: "Revoke access" });
    expect(screen.queryByRole("button", { name: "Add worker" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Get install command" })).not.toBeInTheDocument();
    await user.click(revoke);
    expect(apiClient.delete).toHaveBeenCalledWith("/lens/workers/worker", { accessToken: "admin" });
    expect(await screen.findByRole("button", { name: "Get install command" })).toBeDisabled();
    expect(screen.getByRole("combobox", { name: "Analysis model" })).toBeVisible();
    expect(listCalls()).toBe(2);
  });
  it("cleans up a newly created key when registration fails before retrying", async () => {
    const user = userEvent.setup();
    vi.mocked(apiClient.get).mockImplementation(async (path) => {
      if (path === "/models") return { data: [{ id: "analysis-model" }] };
      if (path === "/lens") return { workers: [] };
      return { keys: [], total_pages: 0 };
    });
    vi.mocked(apiClient.post)
      .mockResolvedValueOnce({ token_id: "limited-key-id", key: "sk-secret-not-displayed" })
      .mockRejectedValueOnce(new Error("Registration unavailable"))
      .mockResolvedValueOnce({})
      .mockResolvedValueOnce({ token_id: "retry-key-id" })
      .mockResolvedValueOnce(created);
    renderWithLens(<WorkerSettings workers={[]} />, { accessToken: "admin" });
    expect(screen.getByRole("button", { name: "Get install command" })).toBeDisabled();
    expect(screen.getByRole("textbox", { name: "LiteLLM proxy URL", hidden: true })).not.toBeVisible();
    await user.click(screen.getByRole("combobox", { name: "Analysis model" }));
    await user.click(await screen.findByRole("option", { name: "analysis-model" }));
    await user.clear(screen.getByLabelText("Monthly limit (USD)"));
    await user.type(screen.getByLabelText("Monthly limit (USD)"), "12");
    await user.click(screen.getByRole("button", { name: "Get install command" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Registration unavailable");
    expect(apiClient.post).toHaveBeenNthCalledWith(1, "/key/generate", {
      accessToken: "admin",
      body: {
        key_alias: "Lens analysis",
        models: ["analysis-model"],
        max_budget: 12,
        budget_duration: "1mo",
        metadata: { purpose: "lens" },
      },
    });
    await user.click(screen.getByRole("button", { name: "Get install command" }));
    expect(await screen.findByRole("status")).toHaveTextContent("Waiting for your worker");
    expect(apiClient.post).toHaveBeenCalledWith("/key/delete", {
      accessToken: "admin",
      body: { keys: ["limited-key-id"] },
    });
    expect(apiClient.post).toHaveBeenCalledTimes(5);
    expect(apiClient.post).toHaveBeenLastCalledWith("/lens/workers/register", {
      accessToken: "admin",
      body: { name: "Lens worker", analysis_key_id: "retry-key-id" },
    });
    expect(screen.queryByText("sk-secret-not-displayed")).not.toBeInTheDocument();
  });
});
