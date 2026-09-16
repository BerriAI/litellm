import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders } from "@/../tests/test-utils";
import { getClaudeCodePluginsList, deleteClaudeCodePlugin } from "@/components/networking";
import type { Plugin } from "@/components/claude_code_plugins/types";

import ClaudeCodePluginsPanel from "./ClaudeCodePluginsPanel";

vi.mock("@/components/networking", () => ({
  getClaudeCodePluginsList: vi.fn(),
  deleteClaudeCodePlugin: vi.fn(),
}));

vi.mock("./PluginTable", () => ({
  __esModule: true,
  default: ({
    isLoading,
    pluginsList,
    onDeleteClick,
    onPluginClick,
  }: {
    isLoading: boolean;
    pluginsList: Plugin[];
    onDeleteClick: (pluginName: string, displayName: string) => void;
    onPluginClick: (pluginId: string) => void;
  }) => (
    <div data-testid="plugin-table">
      {isLoading ? "table-loading" : "table-loaded"}
      {pluginsList.map((plugin) => (
        <button
          key={plugin.id}
          data-testid={`row-delete-${plugin.id}`}
          onClick={() => onDeleteClick(plugin.name, plugin.name)}
        >
          row delete
        </button>
      ))}
      {pluginsList.map((plugin) => (
        <button key={`open-${plugin.id}`} onClick={() => onPluginClick(plugin.id)}>
          open {plugin.name}
        </button>
      ))}
    </div>
  ),
}));

vi.mock("./add_plugin_form", () => ({ __esModule: true, default: () => null }));
vi.mock("@/components/claude_code_plugins/skill_detail", () => ({
  __esModule: true,
  default: ({ skill, onBack }: { skill: Plugin; onBack: () => void }) => (
    <div>
      <p data-testid="skill-detail">{skill.name}</p>
      <button onClick={onBack}>Back from detail</button>
    </div>
  ),
}));

const mockGetClaudeCodePluginsList = vi.mocked(getClaudeCodePluginsList);
const mockDeleteClaudeCodePlugin = vi.mocked(deleteClaudeCodePlugin);

const skill: Plugin = {
  id: "plugin-1",
  name: "my-skill",
  source: { source: "github", repo: "acme/my-skill" },
  enabled: true,
};

describe("ClaudeCodePluginsPanel loading state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should resolve the loading state when accessToken is null instead of showing the skeleton forever", async () => {
    renderWithProviders(<ClaudeCodePluginsPanel accessToken={null} />);
    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    expect(mockGetClaudeCodePluginsList).not.toHaveBeenCalled();
  });

  it("should show the loading state until the skills fetch settles", async () => {
    let resolveFetch: (value: { plugins: never[]; count: number }) => void = () => {};
    mockGetClaudeCodePluginsList.mockReturnValue(
      new Promise((resolve) => {
        resolveFetch = resolve;
      }),
    );
    renderWithProviders(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />);
    expect(screen.getByText("table-loading")).toBeInTheDocument();

    resolveFetch({ plugins: [], count: 0 });
    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    expect(mockGetClaudeCodePluginsList).toHaveBeenCalledWith("sk-test", false);
  });
});

describe("ClaudeCodePluginsPanel delete confirmation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetClaudeCodePluginsList.mockResolvedValue({ plugins: [skill], count: 1 });
  });

  it("should ask for confirmation before deleting and name the skill", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />);

    await user.click(await screen.findByTestId("row-delete-plugin-1"));

    expect(await screen.findByText(/are you sure you want to delete skill/i)).toBeInTheDocument();
    expect(screen.getByText("my-skill")).toBeInTheDocument();
    expect(screen.getByText("This action cannot be undone.")).toBeInTheDocument();
    expect(mockDeleteClaudeCodePlugin).not.toHaveBeenCalled();
  });

  it("should delete the skill and refresh the list once confirmed", async () => {
    const user = userEvent.setup();
    mockDeleteClaudeCodePlugin.mockResolvedValue({});
    renderWithProviders(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />);

    await user.click(await screen.findByTestId("row-delete-plugin-1"));
    await screen.findByText(/are you sure you want to delete skill/i);
    await user.click(screen.getByRole("button", { name: "Delete" }));

    await waitFor(() => expect(mockDeleteClaudeCodePlugin).toHaveBeenCalledWith("sk-test", "my-skill"));
    await waitFor(() => expect(mockGetClaudeCodePluginsList).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByText(/are you sure you want to delete skill/i)).not.toBeInTheDocument());
  });

  it("should not delete the skill when the confirmation is cancelled", async () => {
    const user = userEvent.setup();
    renderWithProviders(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />);

    await user.click(await screen.findByTestId("row-delete-plugin-1"));
    await screen.findByText(/are you sure you want to delete skill/i);
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    await waitFor(() => expect(screen.queryByText(/are you sure you want to delete skill/i)).not.toBeInTheDocument());
    expect(mockDeleteClaudeCodePlugin).not.toHaveBeenCalled();
  });
});

describe("ClaudeCodePluginsPanel selected skill URL state", () => {
  const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) =>
    onUrlUpdate.mock.calls.at(-1)?.[0];

  beforeEach(() => {
    vi.clearAllMocks();
    mockGetClaudeCodePluginsList.mockResolvedValue({ plugins: [skill], count: 1 });
  });

  it("opens the skill named in the URL once the list loads", async () => {
    let resolveFetch: (value: { plugins: Plugin[]; count: number }) => void = () => {};
    mockGetClaudeCodePluginsList.mockReturnValue(
      new Promise((resolve) => {
        resolveFetch = resolve;
      }),
    );
    renderWithProviders(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />, {
      searchParams: "?skill=plugin-1",
    });
    expect(screen.getByText("Loading skill…")).toBeInTheDocument();
    expect(screen.queryByTestId("plugin-table")).not.toBeInTheDocument();

    resolveFetch({ plugins: [skill], count: 1 });

    expect(await screen.findByTestId("skill-detail")).toHaveTextContent("my-skill");
    expect(screen.queryByText("Loading skill…")).not.toBeInTheDocument();
  });

  it("shows a not-found state for an unknown skill and clears skill on back", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />, {
      searchParams: "?skill=plugin-gone",
      onUrlUpdate,
    });

    expect(await screen.findByText("Skill not found")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Back to Skills" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("skill")).toBe(false));
    expect(await screen.findByTestId("plugin-table")).toBeInTheDocument();
  });

  it("pushes skill when a row is opened and clears it when the detail closes", async () => {
    const user = userEvent.setup();
    const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
    renderWithProviders(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />, { onUrlUpdate });

    await user.click(await screen.findByRole("button", { name: "open my-skill" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.get("skill")).toBe("plugin-1"));
    expect(lastUrlUpdate(onUrlUpdate)?.options.history).toBe("push");
    expect(screen.getByTestId("skill-detail")).toHaveTextContent("my-skill");

    await user.click(screen.getByRole("button", { name: "Back from detail" }));

    await waitFor(() => expect(lastUrlUpdate(onUrlUpdate)?.searchParams.has("skill")).toBe(false));
    expect(screen.getByTestId("plugin-table")).toBeInTheDocument();
    expect(screen.queryByTestId("skill-detail")).not.toBeInTheDocument();
  });
});
