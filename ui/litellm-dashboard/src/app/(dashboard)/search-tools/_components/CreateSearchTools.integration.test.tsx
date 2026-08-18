import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, testQueryClient } from "@/../tests/test-utils";
import CreateSearchTool from "./CreateSearchTools";

vi.mock("@/components/networking", () => ({
  createSearchTool: vi.fn(),
  fetchAvailableSearchProviders: vi.fn(async () => ({
    providers: [
      { provider_name: "perplexity", ui_friendly_name: "Perplexity AI" },
      { provider_name: "searxng", ui_friendly_name: "SearXNG" },
    ],
  })),
}));

const renderModal = () =>
  renderWithProviders(
    <CreateSearchTool
      userRole="Admin"
      accessToken="sk-test"
      onCreateSuccess={vi.fn()}
      isModalVisible
      setModalVisible={vi.fn()}
    />,
  );

const openProviderDropdown = async (user: ReturnType<typeof userEvent.setup>) => {
  await user.click(await screen.findByRole("combobox"));
  return screen.getByRole("listbox").parentElement as HTMLElement;
};

describe("CreateSearchTool provider dropdown", () => {
  beforeEach(() => {
    testQueryClient.clear();
  });

  it("keeps a provider listed when its slug is typed into the search box", async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    renderModal();

    const dropdown = await openProviderDropdown(user);
    expect(await within(dropdown).findByTitle("SearXNG")).toBeInTheDocument();

    await user.type(screen.getByRole("combobox"), "searxng");

    expect(within(dropdown).getByTitle("SearXNG")).toBeInTheDocument();
    expect(within(dropdown).queryByTitle("Perplexity AI")).not.toBeInTheDocument();
  });

  it("keeps a provider listed when its display name is typed into the search box", async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    renderModal();

    const dropdown = await openProviderDropdown(user);
    await within(dropdown).findByTitle("Perplexity AI");

    await user.type(screen.getByRole("combobox"), "perplexity ai");

    expect(within(dropdown).getByTitle("Perplexity AI")).toBeInTheDocument();
    expect(within(dropdown).queryByTitle("SearXNG")).not.toBeInTheDocument();
  });

  it("shows the provider logo in the dropdown options", async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 });
    renderModal();

    const dropdown = await openProviderDropdown(user);
    expect(await within(dropdown).findByRole("img", { name: "Perplexity AI logo" })).toHaveAttribute(
      "src",
      expect.stringContaining("perplexity.png"),
    );
  });
});
