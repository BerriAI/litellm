import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { apiClient } from "@/components/networking";
import { WorkerDialog } from "./WorkerDialog";

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
    renderWithProviders(<WorkerDialog accessToken="admin" workers={[]} onClose={vi.fn()} onChanged={vi.fn()} />);
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
    expect(screen.getByRole("button", { name: "Token copied" })).toBeVisible();
  });
  it("assigns billing to an existing worker without replacing its access token", async () => {
    const user = userEvent.setup();
    const changed = vi.fn();
    vi.mocked(apiClient.put).mockResolvedValue(created.worker);
    renderWithProviders(
      <WorkerDialog
        accessToken="admin"
        workers={[{ ...created.worker, analysis_key_id: null }]}
        onClose={vi.fn()}
        onChanged={changed}
      />,
    );
    expect(screen.getByText("Billing key required")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Settings" }));
    await user.click(screen.getByRole("combobox", { name: "Charge analysis to" }));
    await user.click(await screen.findByRole("option", { name: "Analysis" }));
    await user.click(screen.getByRole("button", { name: "Save analysis access" }));
    expect(apiClient.put).toHaveBeenCalledWith("/lens/workers/worker/billing-key", {
      accessToken: "admin",
      body: { analysis_key_id: "b".repeat(64) },
    });
    expect(changed).toHaveBeenCalledOnce();
    expect(apiClient.post).not.toHaveBeenCalled();
  });
  it("requires revoking the current worker before setting up a replacement", async () => {
    const user = userEvent.setup();
    const changed = vi.fn();
    vi.mocked(apiClient.delete).mockResolvedValue(true);
    const props = { accessToken: "admin", onClose: vi.fn(), onChanged: changed };
    const view = renderWithProviders(<WorkerDialog {...props} workers={[created.worker]} />);
    expect(screen.queryByRole("button", { name: "Add worker" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Get install command" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Revoke access" }));
    expect(apiClient.delete).toHaveBeenCalledWith("/lens/workers/worker", { accessToken: "admin" });
    expect(changed).toHaveBeenCalledOnce();
    view.rerender(<WorkerDialog {...props} workers={[{ ...created.worker, revoked: true }]} />);
    expect(screen.getByRole("button", { name: "Get install command" })).toBeDisabled();
    expect(screen.getByRole("combobox", { name: "Analysis model" })).toBeVisible();
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
    renderWithProviders(<WorkerDialog accessToken="admin" workers={[]} onClose={vi.fn()} onChanged={vi.fn()} />);
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
