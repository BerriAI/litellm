import { fireEvent, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, testQueryClient } from "../../../tests/test-utils";
import DiscoverPage from "./DiscoverPage";
import { LAUNCHES } from "./discoverContent";

const mockModelCostMap = vi.fn();

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-test", userRole: "Admin", premiumUser: false, isViewOnly: false }),
}));

vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  modelCostMap: (...args: unknown[]) => mockModelCostMap(...args),
}));

const costMap = {
  "gpt-6.1-sol": { litellm_provider: "openai", mode: "chat", max_input_tokens: 922_000, input_cost_per_token: 2e-6 },
  "claude-haiku-5-5": { litellm_provider: "anthropic", mode: "chat", max_input_tokens: 1_000_000 },
  "whisper-1": { litellm_provider: "openai", mode: "audio_transcription" },
};

describe("DiscoverPage", () => {
  beforeEach(() => {
    testQueryClient.clear();
    mockModelCostMap.mockReset();
    mockModelCostMap.mockResolvedValue(costMap);
  });

  it("lists every launch newest first with a new-tab link", async () => {
    renderWithProviders(<DiscoverPage />);
    const section = (await screen.findByRole("heading", { name: "What's new" })).closest("section")!;
    const links = within(section).getAllByRole("link");
    expect(links.map((l) => l.textContent)).toEqual(
      [...LAUNCHES].sort((a, b) => b.publishedOn.localeCompare(a.publishedOn)).map((l) => expect.stringContaining(l.title)),
    );
    links.forEach((link) => {
      expect(link).toHaveAttribute("target", "_blank");
      expect(link).toHaveAttribute("rel", "noopener noreferrer");
    });
  });

  it("filters launches by kind and writes the filter to the URL", async () => {
    const onUrlUpdate = vi.fn();
    const user = userEvent.setup();
    renderWithProviders(<DiscoverPage />, { onUrlUpdate });
    await user.click(screen.getByRole("button", { name: "Providers" }));
    const section = screen.getByRole("heading", { name: "What's new" }).closest("section")!;
    expect(within(section).getAllByText("New provider")).toHaveLength(
      LAUNCHES.filter((l) => l.kind === "provider").length,
    );
    expect(within(section).queryByText("New model")).not.toBeInTheDocument();
    expect(onUrlUpdate.mock.lastCall?.[0].searchParams.get("kind")).toBe("provider");
  });

  it("loads the catalog-only cost map and shows the featured models with pricing", async () => {
    renderWithProviders(<DiscoverPage />);
    const list = (await screen.findByRole("heading", { name: "Discover models" })).closest("section")!;
    expect(await within(list).findByText("gpt-6.1-sol")).toBeInTheDocument();
    expect(mockModelCostMap).toHaveBeenCalledWith(true);
    expect(within(list).getByText("3 in the LiteLLM catalog")).toBeInTheDocument();
    const items = within(list).getAllByRole("listitem");
    expect(items.map((i) => within(i).getByTitle(/.+/).textContent)).toEqual(["gpt-6.1-sol", "claude-haiku-5-5"]);
    expect(within(items[0]).getByText("$2.00")).toBeInTheDocument();
    expect(within(items[0]).getByRole("link", { name: /Add to proxy/ })).toHaveAttribute(
      "href",
      expect.stringContaining("models-and-endpoints?tab=add"),
    );
  });

  it("searches the whole catalog by name and syncs the query to the URL", async () => {
    const onUrlUpdate = vi.fn();
    renderWithProviders(<DiscoverPage />, { onUrlUpdate });
    await screen.findByText("gpt-6.1-sol");
    fireEvent.change(screen.getByRole("textbox", { name: "Search models" }), { target: { value: "whisper" } });
    expect(await screen.findByText("whisper-1")).toBeInTheDocument();
    expect(screen.queryByText("gpt-6.1-sol")).not.toBeInTheDocument();
    expect(onUrlUpdate.mock.lastCall?.[0].searchParams.get("q")).toBe("whisper");
  });

  it("opens with the provider filter from the URL applied", async () => {
    renderWithProviders(<DiscoverPage />, { searchParams: "?provider=anthropic" });
    expect(await screen.findByText("claude-haiku-5-5")).toBeInTheDocument();
    expect(screen.queryByText("gpt-6.1-sol")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Anthropic" })).toHaveAttribute("aria-pressed", "true");
  });
});
