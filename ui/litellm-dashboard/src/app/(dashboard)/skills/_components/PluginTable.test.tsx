import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { SkillHubSkill } from "@/components/claude_code_plugins/types";

import PluginTable from "./PluginTable";

const mockPlugins: SkillHubSkill[] = [
  {
    id: "plugin-id-newer",
    name: "newer-skill",
    version: "1.2.0",
    description: "A skill for testing",
    source: { source: "github", repo: "org/newer-skill" },
    domain: "development",
    visibility: "public",
    updated_at: "2025-01-15T10:30:00Z",
  },
  {
    id: "plugin-id-older",
    name: "older-skill",
    source: { source: "github", repo: "org/older-skill" },
    visibility: "private",
    updated_at: "2024-01-10T09:15:00Z",
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

  it.each([
    [null, "—"],
    [0, "0"],
    [12345, "12,345"],
  ] as const)("renders headers and download count %s", (download_count, display) => {
    render(<PluginTable {...defaultProps} pluginsList={[{ ...mockPlugins[0], download_count }]} />);
    for (const header of ["Skill Name", "Version", "Description", "Domain", "Public", "Updated at", "Downloads"]) {
      expect(screen.getByText(header)).toBeInTheDocument();
    }
    expect(within(screen.getByRole("row", { name: /newer-skill/ })).getByText(display)).toBeInTheDocument();
  });

  it("should display the empty state when data is empty", () => {
    render(<PluginTable {...defaultProps} pluginsList={[]} />);
    expect(screen.getByText("No skills found")).toBeInTheDocument();
  });

  it("should sort by name ascending by default", () => {
    render(<PluginTable {...defaultProps} />);
    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("newer-skill")).toBeInTheDocument();
    expect(within(rows[1]).getByText("older-skill")).toBeInTheDocument();
  });

  it("should call onPluginClick with the plugin ID when the skill name is clicked", async () => {
    const user = userEvent.setup();
    render(<PluginTable {...defaultProps} />);
    await user.click(screen.getByRole("button", { name: "newer-skill" }));
    expect(mockOnPluginClick).toHaveBeenCalledWith("plugin-id-newer");
  });

  it("should not navigate when clicking elsewhere in the row", async () => {
    const user = userEvent.setup();
    render(<PluginTable {...defaultProps} />);
    await user.click(screen.getByText("A skill for testing"));
    expect(mockOnPluginClick).not.toHaveBeenCalled();
  });

  it("should display the domain and a fallback when unset", () => {
    render(<PluginTable {...defaultProps} />);
    expect(screen.getByText("development")).toBeInTheDocument();
    expect(screen.getAllByText("—")[0]).toBeInTheDocument();
  });

  it("should show whether the skill is public", () => {
    render(<PluginTable {...defaultProps} />);
    expect(screen.getByText("Yes")).toBeInTheDocument();
    expect(screen.getByText("No")).toBeInTheDocument();
  });

  it("should delete a skill through the actions menu when admin", async () => {
    const user = userEvent.setup();
    render(<PluginTable {...defaultProps} />);
    await user.click(screen.getByTestId("plugin-actions-newer-skill"));
    await user.click(await screen.findByTestId("plugin-action-delete"));
    expect(mockOnDeleteClick).toHaveBeenCalledWith("newer-skill", "newer-skill");
  });

  it("should copy the skill ID through the actions menu", async () => {
    const user = userEvent.setup();
    render(<PluginTable {...defaultProps} />);
    await user.click(screen.getByTestId("plugin-actions-newer-skill"));
    await user.click(await screen.findByTestId("plugin-action-copy"));
    expect(await window.navigator.clipboard.readText()).toBe("plugin-id-newer");
  });

  it("should hide the delete action for non-admins but keep copy available", async () => {
    const user = userEvent.setup();
    render(<PluginTable {...defaultProps} isAdmin={false} />);
    expect(screen.queryByRole("columnheader", { name: "Downloads" })).not.toBeInTheDocument();
    await user.click(screen.getByTestId("plugin-actions-newer-skill"));
    expect(await screen.findByTestId("plugin-action-copy")).toBeInTheDocument();
    expect(screen.queryByTestId("plugin-action-delete")).not.toBeInTheDocument();
  });
});
