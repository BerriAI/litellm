import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { apiClient } from "@/components/networking";
import { WorkerSetup } from "./WorkerSetup";

vi.mock("@/components/networking", () => ({
  apiClient: { get: vi.fn(), post: vi.fn(), put: vi.fn() },
  proxyBaseUrl: "https://gateway.example/proxy",
}));

const created = {
  token: "lens-test-token",
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
    vi.mocked(apiClient.get).mockResolvedValue({
      keys: [{ token: "b".repeat(64), key_alias: "Analysis" }],
      total_pages: 1,
    });
  });
  it("generates a complete command using one worker credential and the configured proxy address", async () => {
    vi.mocked(apiClient.post).mockResolvedValue(created);
    const user = userEvent.setup();
    renderWithProviders(<WorkerSetup accessToken="admin" workers={[]} onClose={vi.fn()} onChanged={vi.fn()} />);
    expect(screen.getByRole("textbox", { name: "Your LiteLLM deployment URL" })).toHaveValue(
      "https://gateway.example/proxy",
    );
    expect(screen.getByRole("button", { name: "Generate setup command" })).toBeDisabled();
    await user.click(screen.getByRole("combobox", { name: "Charge analysis to" }));
    await user.click(await screen.findByRole("option", { name: "Analysis" }));
    await user.click(screen.getByRole("button", { name: "Generate setup command" }));
    expect(apiClient.post).toHaveBeenCalledWith("/engine/workers/register", {
      accessToken: "admin",
      body: { name: "Lens analyzer", analysis_key_id: "b".repeat(64) },
    });
    expect(screen.getByRole("status")).toHaveTextContent("Waiting for your analyzer to connect");
    await user.click(screen.getByRole("button", { name: "Copy Docker command" }));
    const command = await navigator.clipboard.readText();
    expect(command).toContain("LITELLM_URL=https://gateway.example/proxy");
    expect(command).toContain("LENS_WORKER_TOKEN=lens-test-token");
    expect(command).toContain("--add-host host.docker.internal:host-gateway");
    expect(command).toContain("ghcr.io/berriai/litellm-lens-worker@sha256:");
  });
  it("assigns billing to an existing worker without replacing its access token", async () => {
    const user = userEvent.setup();
    const changed = vi.fn();
    vi.mocked(apiClient.put).mockResolvedValue(created.worker);
    renderWithProviders(
      <WorkerSetup
        accessToken="admin"
        workers={[{ ...created.worker, analysis_key_id: null }]}
        onClose={vi.fn()}
        onChanged={changed}
      />,
    );
    expect(screen.getByText("Billing key required")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Billing key" }));
    await user.click(screen.getByRole("combobox", { name: "Charge analysis to" }));
    await user.click(await screen.findByRole("option", { name: "Analysis" }));
    await user.click(screen.getByRole("button", { name: "Save billing key" }));
    expect(apiClient.put).toHaveBeenCalledWith("/engine/workers/worker/billing-key", {
      accessToken: "admin",
      body: { analysis_key_id: "b".repeat(64) },
    });
    expect(changed).toHaveBeenCalledOnce();
    expect(apiClient.post).not.toHaveBeenCalled();
  });
});
