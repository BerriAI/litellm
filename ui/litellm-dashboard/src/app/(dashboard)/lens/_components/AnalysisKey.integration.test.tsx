import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import { apiClient } from "@/components/networking";
import { AnalysisKey } from "./AnalysisKey";

vi.mock("@/components/networking", () => ({ apiClient: { get: vi.fn(), post: vi.fn() } }));

describe("Lens billing key", () => {
  beforeEach(() => {
    testQueryClient.clear();
    vi.clearAllMocks();
  });
  it("creates a normal key and only passes its ID to worker settings", async () => {
    const user = userEvent.setup();
    const changed = vi.fn();
    vi.mocked(apiClient.get).mockResolvedValue({ keys: [], total_pages: 0 });
    vi.mocked(apiClient.post).mockResolvedValue({ token_id: "b".repeat(64), key: "sk-secret-not-for-settings" });
    renderWithProviders(<AnalysisKey accessToken="test" value={null} onChange={changed} name="Research" />);
    await user.click(screen.getByRole("button", { name: "Create worker key" }));
    expect(await screen.findByRole("combobox", { name: "Charge analysis to" })).toHaveValue("Lens: Research");
    expect(apiClient.post).toHaveBeenCalledWith("/key/generate", {
      accessToken: "test",
      body: { key_alias: "Lens: Research", models: [], metadata: { purpose: "lens" } },
    });
    expect(changed).toHaveBeenCalledExactlyOnceWith("b".repeat(64));
    expect(screen.queryByText("sk-secret-not-for-settings")).not.toBeInTheDocument();
  });

  it("pages existing keys without dropping the selected billing key", async () => {
    const user = userEvent.setup();
    const changed = vi.fn();
    vi.mocked(apiClient.get).mockImplementation(async (_path, options) => ({
      keys:
        options?.query?.page === "2"
          ? [{ token: "c".repeat(64), key_alias: "Second page" }]
          : [{ token: "a".repeat(64), key_alias: "First page" }],
      total_pages: 2,
    }));
    renderWithProviders(<AnalysisKey accessToken="test" value={null} onChange={changed} name="Research" />);
    await user.click(screen.getByRole("combobox", { name: "Charge analysis to" }));
    await user.click(await screen.findByRole("option", { name: "Load more keys" }));
    await user.click(await screen.findByRole("option", { name: "Second page" }));
    expect(changed).toHaveBeenCalledExactlyOnceWith("c".repeat(64));
  });
});
