import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "@/../tests/test-utils";
import { apiClient } from "@/components/networking";
import { WorkerSetup } from "./WorkerSetup";

vi.mock("@/components/networking", () => ({
  apiClient: { post: vi.fn() },
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
  },
};

describe("Worker setup", () => {
  it("generates a complete command using one worker credential and the configured proxy address", async () => {
    vi.mocked(apiClient.post).mockResolvedValue(created);
    const user = userEvent.setup();
    renderWithProviders(<WorkerSetup accessToken="admin" workers={[]} onClose={vi.fn()} onChanged={vi.fn()} />);
    expect(screen.getByRole("textbox", { name: "Your LiteLLM deployment URL" })).toHaveValue("https://gateway.example/proxy");
    await user.click(screen.getByRole("button", { name: "Generate setup command" }));
    expect(screen.getByRole("status")).toHaveTextContent("Waiting for your analyzer to connect");
    await user.click(screen.getByRole("button", { name: "Copy Docker command" }));
    const command = await navigator.clipboard.readText();
    expect(command).toContain("LITELLM_URL=https://gateway.example/proxy");
    expect(command).toContain("LENS_WORKER_TOKEN=lens-test-token");
    expect(command).toContain("ghcr.io/berriai/litellm-lens-worker@sha256:");
  });
});
