import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen } from "../../../tests/test-utils";
import { fetchSearchTools } from "../networking";
import SearchToolSelector from "./SearchToolSelector";

vi.mock("../networking", () => ({
  fetchSearchTools: vi.fn(),
}));

describe("SearchToolSelector", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchSearchTools).mockResolvedValue({
      search_tools: [{ search_tool_name: "search-one" }, { search_tool_name: "search-two" }],
    });
  });

  it("should render", () => {
    renderWithProviders(<SearchToolSelector accessToken="" onChange={vi.fn()} />);

    expect(screen.getByRole("combobox")).toBeInTheDocument();
  });

  it("should load and display available search tools", async () => {
    const user = userEvent.setup();
    renderWithProviders(<SearchToolSelector accessToken="token" onChange={vi.fn()} />);

    await user.click(screen.getByRole("combobox"));

    expect(await screen.findByRole("option", { name: "search-one" })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: "search-two" })).toBeInTheDocument();
  });
});
