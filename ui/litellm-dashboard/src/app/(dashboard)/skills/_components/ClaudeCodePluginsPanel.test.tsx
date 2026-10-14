import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getClaudeCodePluginsList, deleteClaudeCodePlugin } from "@/components/networking";
import type { Plugin } from "@/components/claude_code_plugins/types";

import ClaudeCodePluginsPanel from "./ClaudeCodePluginsPanel";

vi.mock("@/components/networking", () => ({
  getClaudeCodePluginsList: vi.fn(),
  deleteClaudeCodePlugin: vi.fn(),
  getProxyBaseUrl: vi.fn(() => "https://proxy.example.com/root/"),
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
    onPluginClick: (id: string) => void;
  }) => (
    <div data-testid="plugin-table">
      {isLoading ? "table-loading" : "table-loaded"}
      {pluginsList.map((plugin) => (
        <div key={plugin.id}>
          <button data-testid={`row-delete-${plugin.id}`} onClick={() => onDeleteClick(plugin.name, plugin.name)}>
            row delete
          </button>
          <button data-testid={`row-open-${plugin.id}`} onClick={() => onPluginClick(plugin.id)}>
            Open plugin
          </button>
        </div>
      ))}
    </div>
  ),
}));

vi.mock("./add_plugin_form", () => ({ __esModule: true, default: () => null }));

const mockGetClaudeCodePluginsList = vi.mocked(getClaudeCodePluginsList);
const mockDeleteClaudeCodePlugin = vi.mocked(deleteClaudeCodePlugin);

const skill: Plugin = {
  id: "plugin-1",
  name: "my-skill",
  source: { source: "github", repo: "acme/my-skill" },
  enabled: true,
};
const hostedSkill: Plugin = { ...skill, source: { source: "hosted" } };

describe("ClaudeCodePluginsPanel loading state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should resolve the loading state when accessToken is null instead of showing the skeleton forever", async () => {
    render(<ClaudeCodePluginsPanel accessToken={null} />);
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
    render(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />);
    expect(screen.getByText("table-loading")).toBeInTheDocument();

    resolveFetch({ plugins: [], count: 0 });
    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    expect(mockGetClaudeCodePluginsList).toHaveBeenCalledWith("sk-test", false);
  });

  it("shows the Claude Desktop marketplace URL and configuration", async () => {
    mockGetClaudeCodePluginsList.mockResolvedValue({ plugins: [], count: 0 });
    render(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />);

    expect(await screen.findByText("https://proxy.example.com/root/claude-code/marketplace.json")).toBeInTheDocument();
    expect(screen.getByText("Connect Claude Desktop")).toBeInTheDocument();
    expect(screen.getByText((_, element) => element?.tagName === "PRE")).toHaveTextContent(
      '"credentialKind": "inferenceCredential"',
    );
    expect(screen.getByRole("button", { name: "Copy JSON" })).toBeInTheDocument();
  });

  it("shows hosted plugins as hosted rather than a URL", async () => {
    mockGetClaudeCodePluginsList.mockResolvedValue({ plugins: [hostedSkill], count: 1 });
    render(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />);

    fireEvent.click(await screen.findByTestId("row-open-plugin-1"));

    expect(await screen.findByText("Hosted on LiteLLM")).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });
});

describe("ClaudeCodePluginsPanel delete confirmation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetClaudeCodePluginsList.mockResolvedValue({ plugins: [skill], count: 1 });
  });

  it("should ask for confirmation before deleting and name the skill", async () => {
    const user = userEvent.setup();
    render(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />);

    await user.click(await screen.findByTestId("row-delete-plugin-1"));

    expect(await screen.findByText(/are you sure you want to delete skill/i)).toBeInTheDocument();
    expect(screen.getByText("my-skill")).toBeInTheDocument();
    expect(screen.getByText("This action cannot be undone.")).toBeInTheDocument();
    expect(mockDeleteClaudeCodePlugin).not.toHaveBeenCalled();
  });

  it("should delete the skill and refresh the list once confirmed", async () => {
    const user = userEvent.setup();
    mockDeleteClaudeCodePlugin.mockResolvedValue({});
    render(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />);

    await user.click(await screen.findByTestId("row-delete-plugin-1"));
    await screen.findByText(/are you sure you want to delete skill/i);
    await user.click(screen.getByRole("button", { name: "Delete" }));

    await waitFor(() => expect(mockDeleteClaudeCodePlugin).toHaveBeenCalledWith("sk-test", "my-skill"));
    await waitFor(() => expect(mockGetClaudeCodePluginsList).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByText(/are you sure you want to delete skill/i)).not.toBeInTheDocument());
  });

  it("should not delete the skill when the confirmation is cancelled", async () => {
    const user = userEvent.setup();
    render(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />);

    await user.click(await screen.findByTestId("row-delete-plugin-1"));
    await screen.findByText(/are you sure you want to delete skill/i);
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    await waitFor(() => expect(screen.queryByText(/are you sure you want to delete skill/i)).not.toBeInTheDocument());
    expect(mockDeleteClaudeCodePlugin).not.toHaveBeenCalled();
  });
});
