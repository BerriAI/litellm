import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/../tests/test-utils";
import { Plugin } from "@/components/claude_code_plugins/types";

import PluginTable from "./PluginTable";

const mockPlugins: Plugin[] = [
  {
    id: "plugin-id-newer",
    name: "newer-skill",
    version: "1.2.0",
    description: "A skill for testing",
    source: { source: "github", repo: "org/newer-skill" },
    category: "development",
    enabled: true,
    created_at: "2025-01-15T10:30:00Z",
  },
  {
    id: "plugin-id-older",
    name: "older-skill",
    source: { source: "github", repo: "org/older-skill" },
    enabled: false,
    created_at: "2024-01-10T09:15:00Z",
  },
];

const mockOnDeleteClick = vi.fn();
const mockOnPluginClick = vi.fn();

const defaultProps = {
  pluginsList: mockPlugins,
  isLoading: false,
  onDeleteClick: mockOnDeleteClick,
  isAdmin: true,
  onPluginClick: mockOnPluginClick,
};

describe("PluginTable", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render every column header", () => {
    renderWithProviders(<PluginTable {...defaultProps} />);
    for (const header of ["Skill Name", "Version", "Description", "Category", "Public", "Created At"]) {
      expect(screen.getByText(header)).toBeInTheDocument();
    }
  });

  it("should display the empty state when data is empty", () => {
    renderWithProviders(<PluginTable {...defaultProps} pluginsList={[]} />);
    expect(screen.getByText("No skills found")).toBeInTheDocument();
  });

  it("should sort by created date descending by default", () => {
    renderWithProviders(<PluginTable {...defaultProps} />);
    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("newer-skill")).toBeInTheDocument();
    expect(within(rows[1]).getByText("older-skill")).toBeInTheDocument();
  });

  it("should call onPluginClick with the plugin ID when the skill name is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PluginTable {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: "newer-skill" }));
    expect(mockOnPluginClick).toHaveBeenCalledWith("plugin-id-newer");
  });

  it("should not navigate when clicking elsewhere in the row", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PluginTable {...defaultProps} />);
    await user.click(screen.getByText("A skill for testing"));
    expect(mockOnPluginClick).not.toHaveBeenCalled();
  });

  it("should badge the category and fall back to Uncategorized", () => {
    renderWithProviders(<PluginTable {...defaultProps} />);
    expect(screen.getByText("development")).toBeInTheDocument();
    expect(screen.getByText("Uncategorized")).toBeInTheDocument();
  });

  it("should show whether the skill is public", () => {
    renderWithProviders(<PluginTable {...defaultProps} />);
    expect(screen.getByText("Yes")).toBeInTheDocument();
    expect(screen.getByText("No")).toBeInTheDocument();
  });

  it("should delete a skill through the actions menu when admin", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PluginTable {...defaultProps} />);
    await user.click(screen.getByTestId("plugin-actions-newer-skill"));
    await user.click(await screen.findByTestId("plugin-action-delete"));
    expect(mockOnDeleteClick).toHaveBeenCalledWith("newer-skill", "newer-skill");
  });

  it("should copy the skill ID through the actions menu", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PluginTable {...defaultProps} />);
    await user.click(screen.getByTestId("plugin-actions-newer-skill"));
    await user.click(await screen.findByTestId("plugin-action-copy"));
    expect(await window.navigator.clipboard.readText()).toBe("plugin-id-newer");
  });

  it("should hide the delete action for non-admins but keep copy available", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PluginTable {...defaultProps} isAdmin={false} />);
    await user.click(screen.getByTestId("plugin-actions-newer-skill"));
    expect(await screen.findByTestId("plugin-action-copy")).toBeInTheDocument();
    expect(screen.queryByTestId("plugin-action-delete")).not.toBeInTheDocument();
  });

  describe("URL table state", () => {
    const makePlugin = (name: string, createdAt: string): Plugin => ({
      id: `id-${name}`,
      name,
      source: { source: "github", repo: `org/${name}` },
      enabled: true,
      created_at: createdAt,
    });
    const newerAlpha = makePlugin("skill-alpha", "2025-01-01T00:00:00Z");
    const olderZeta = makePlugin("skill-zeta", "2023-01-01T00:00:00Z");
    const manyPlugins = Array.from({ length: 30 }, (_, index) =>
      makePlugin(`skill-${String(index).padStart(2, "0")}`, `2024-01-01T00:00:${String(59 - index).padStart(2, "0")}Z`),
    );
    const rowNames = () =>
      screen
        .getAllByRole("row")
        .slice(1)
        .map((row) => within(row).getByText(/^skill-/).textContent);
    const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
      onUrlUpdate.mock.calls.at(-1)?.[0];

    it("orders rows by the sort in the URL", () => {
      renderWithProviders(<PluginTable {...defaultProps} pluginsList={[olderZeta, newerAlpha]} />, {
        searchParams: "?sort_by=name&sort_order=desc",
      });
      expect(rowNames()).toEqual(["skill-zeta", "skill-alpha"]);
    });

    it("writes the sort to the URL when a header is clicked", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<PluginTable {...defaultProps} pluginsList={[newerAlpha, olderZeta]} />, { onUrlUpdate });

      await user.click(screen.getByTestId("sort-header-created_at"));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("sort_order")).toBe("asc"));
      expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("sort_by")).toBe(false);
      expect(rowNames()).toEqual(["skill-zeta", "skill-alpha"]);
    });

    it("opens the page named in the URL", () => {
      renderWithProviders(<PluginTable {...defaultProps} pluginsList={manyPlugins} />, { searchParams: "?page=2" });
      expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 2 of 2");
      expect(rowNames()).toEqual(["skill-25", "skill-26", "skill-27", "skill-28", "skill-29"]);
    });

    it("writes the page to the URL when paging forward", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<PluginTable {...defaultProps} pluginsList={manyPlugins} />, { onUrlUpdate });

      await user.click(screen.getByRole("button", { name: "Go to next page" }));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("page")).toBe("2"));
      expect(rowNames()[0]).toBe("skill-25");
    });
  });
});
