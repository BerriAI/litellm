import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { getClaudeCodePluginsList, deleteClaudeCodePlugin, reviewClaudeCodePlugin } from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";
import type { Plugin } from "@/components/claude_code_plugins/types";

import ClaudeCodePluginsPanel from "./ClaudeCodePluginsPanel";

vi.mock("@/components/networking", () => ({
  getClaudeCodePluginsList: vi.fn(),
  deleteClaudeCodePlugin: vi.fn(),
  reviewClaudeCodePlugin: vi.fn(),
}));

vi.mock("@/components/molecules/notifications_manager", () => ({
  __esModule: true,
  default: { success: vi.fn(), error: vi.fn(), fromBackend: vi.fn() },
}));

vi.mock("./PluginTable", () => ({
  __esModule: true,
  default: ({
    isLoading,
    pluginsList,
    onDeleteClick,
    onReviewClick,
  }: {
    isLoading: boolean;
    pluginsList: Plugin[];
    onDeleteClick: (pluginName: string, displayName: string) => void;
    onReviewClick: (plugin: Plugin, decision: "approve" | "reject") => void;
  }) => (
    <div data-testid="plugin-table">
      {isLoading ? "table-loading" : "table-loaded"}
      {pluginsList.map((plugin) => (
        <div key={plugin.id}>
          <span>{`row-${plugin.name}`}</span>
          <button data-testid={`row-delete-${plugin.id}`} onClick={() => onDeleteClick(plugin.name, plugin.name)}>
            row delete
          </button>
          <button data-testid={`row-approve-${plugin.id}`} onClick={() => onReviewClick(plugin, "approve")}>
            row approve
          </button>
          <button data-testid={`row-reject-${plugin.id}`} onClick={() => onReviewClick(plugin, "reject")}>
            row reject
          </button>
        </div>
      ))}
    </div>
  ),
}));

vi.mock("./add_plugin_form", () => ({ __esModule: true, default: () => null }));
vi.mock("@/components/claude_code_plugins/skill_detail", () => ({ __esModule: true, default: () => null }));

const mockGetClaudeCodePluginsList = vi.mocked(getClaudeCodePluginsList);
const mockDeleteClaudeCodePlugin = vi.mocked(deleteClaudeCodePlugin);
const mockReviewClaudeCodePlugin = vi.mocked(reviewClaudeCodePlugin);

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

describe("ClaudeCodePluginsPanel review queue", () => {
  const submission: Plugin = {
    id: "plugin-2",
    name: "submitted-skill",
    source: { source: "github", repo: "acme/submitted-skill" },
    enabled: false,
    approval_status: "pending_review",
    manifest_fingerprint: "fingerprint-of-the-reviewed-content",
  };

  beforeEach(() => {
    vi.clearAllMocks();
    mockGetClaudeCodePluginsList.mockResolvedValue({ plugins: [skill, submission], count: 2 });
  });

  it("should let a non-admin submit a skill instead of disabling the button", async () => {
    render(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Internal User" />);
    const submit = await screen.findByRole("button", { name: "+ Submit Skill" });
    expect(submit).toBeEnabled();
    expect(screen.queryByTestId("toggle-pending-review")).not.toBeInTheDocument();
  });

  it("should filter the table down to submissions awaiting review", async () => {
    const user = userEvent.setup();
    render(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />);

    await user.click(await screen.findByTestId("toggle-pending-review"));

    expect(screen.getByText("row-submitted-skill")).toBeInTheDocument();
    expect(screen.queryByText("row-my-skill")).not.toBeInTheDocument();
  });

  it("should approve a submission and refresh the list", async () => {
    const user = userEvent.setup();
    mockReviewClaudeCodePlugin.mockResolvedValue({});
    render(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />);

    await user.click(await screen.findByTestId("row-approve-plugin-2"));
    await user.click(await screen.findByTestId("review-confirm"));

    await waitFor(() =>
      expect(mockReviewClaudeCodePlugin).toHaveBeenCalledWith("sk-test", "submitted-skill", {
        decision: "approve",
        reviewNotes: "",
        reviewedFingerprint: "fingerprint-of-the-reviewed-content",
      }),
    );
    await waitFor(() => expect(mockGetClaudeCodePluginsList).toHaveBeenCalledTimes(2));
  });

  it("should tell the reviewer to look again when the submission changed under them", async () => {
    const user = userEvent.setup();
    mockReviewClaudeCodePlugin.mockRejectedValue(Object.assign(new Error("conflict"), { status: 409 }));
    render(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />);

    await user.click(await screen.findByTestId("row-approve-plugin-2"));
    await user.click(await screen.findByTestId("review-confirm"));

    await waitFor(() =>
      expect(NotificationsManager.error).toHaveBeenCalledWith(
        "This submission changed since the list was loaded. Review the refreshed content before approving it",
      ),
    );
    await waitFor(() => expect(mockGetClaudeCodePluginsList).toHaveBeenCalledTimes(2));
  });

  it("should send the rejection notes typed by the reviewer", async () => {
    const user = userEvent.setup();
    mockReviewClaudeCodePlugin.mockResolvedValue({});
    render(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />);

    await user.click(await screen.findByTestId("row-reject-plugin-2"));
    await user.type(await screen.findByTestId("review-notes"), "point at the skill folder");
    await user.click(screen.getByTestId("review-confirm"));

    await waitFor(() =>
      expect(mockReviewClaudeCodePlugin).toHaveBeenCalledWith("sk-test", "submitted-skill", {
        decision: "reject",
        reviewNotes: "point at the skill folder",
        reviewedFingerprint: "fingerprint-of-the-reviewed-content",
      }),
    );
  });
});
