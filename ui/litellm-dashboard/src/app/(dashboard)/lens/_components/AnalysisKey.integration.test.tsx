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

  it("pages existing keys without dropping the selected billing key", async () => {
    const user = userEvent.setup();
    const changed = vi.fn();
    vi.mocked(apiClient.get).mockImplementation(async (path, options) =>
      path === "/key/info"
        ? { info: { models: ["restricted-model"], max_budget: 4, budget_duration: "1d" } }
        : {
            keys:
              options?.query?.page === "2"
                ? [{ token: "c".repeat(64), key_alias: "Second page" }]
                : [{ token: "a".repeat(64), key_alias: "First page" }],
            total_pages: 2,
          },
    );
    renderWithProviders(<AnalysisKey accessToken="test" value={null} onChange={changed} />);
    await user.click(screen.getByRole("combobox", { name: "Charge analysis to" }));
    await user.click(await screen.findByRole("option", { name: "Load more keys" }));
    await user.click(await screen.findByRole("option", { name: "Second page" }));
    expect(changed).toHaveBeenCalledExactlyOnceWith("c".repeat(64));
    expect(await screen.findByText("restricted-model")).toBeInTheDocument();
    expect(screen.getByText("$4.00 / day")).toBeInTheDocument();
  });
});
