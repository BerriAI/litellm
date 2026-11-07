import { fireEvent, screen, waitFor } from "@testing-library/react";
import { renderWithProviders as render, testQueryClient } from "@/../tests/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { listSkillHubSkills, getSkillHubSkill, deleteSkillHubSkill } from "@/components/networking";
import type { SkillHubSkill } from "@/components/claude_code_plugins/types";

import ClaudeCodePluginsPanel from "./ClaudeCodePluginsPanel";

vi.mock("@/components/networking", () => ({
  listSkillHubSkills: vi.fn(),
  getSkillHubSkill: vi.fn(),
  deleteSkillHubSkill: vi.fn(),
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
    pluginsList: SkillHubSkill[];
    onDeleteClick: (pluginName: string, displayName: string) => void;
    onPluginClick: (id: string) => void;
  }) => (
    <div data-testid="plugin-table">
      {isLoading ? "table-loading" : "table-loaded"}
      {pluginsList.map((plugin) => (
        <button
          key={plugin.id}
          data-testid={`row-delete-${plugin.id}`}
          onDoubleClick={() => onPluginClick(plugin.id)}
          onClick={() => onDeleteClick(plugin.name, plugin.name)}
        >
          row delete
        </button>
      ))}
    </div>
  ),
}));

vi.mock("./add_plugin_form", () => ({ __esModule: true, default: () => null }));
vi.mock("./SkillSubmissions", () => ({ __esModule: true, default: () => null }));
vi.mock("@/components/claude_code_plugins/skill_detail", () => ({
  __esModule: true,
  skillHubDetail: (skill: SkillHubSkill) => skill,
  default: ({ skill, isAdmin }: { skill: SkillHubSkill; isAdmin?: boolean }) => (
    <section aria-label={`Details for ${skill.name}`}>{isAdmin && <button>Downloads</button>}</section>
  ),
}));

const mockListSkillHubSkills = vi.mocked(listSkillHubSkills);
const mockDeleteSkillHubSkill = vi.mocked(deleteSkillHubSkill);

const skill: SkillHubSkill = {
  id: "plugin-1",
  name: "my-skill",
  source: { source: "github", repo: "acme/my-skill" },
  visibility: "public",
};

describe("ClaudeCodePluginsPanel loading state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
  });

  it("should resolve the loading state when accessToken is null instead of showing the skeleton forever", async () => {
    render(<ClaudeCodePluginsPanel accessToken={null} />);
    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    expect(mockListSkillHubSkills).not.toHaveBeenCalled();
  });

  it("should show the loading state until the skills fetch settles", async () => {
    let resolveFetch: (value: { skills: never[]; count: number; next_offset: number | null }) => void = () => {};
    mockListSkillHubSkills.mockReturnValue(
      new Promise((resolve) => {
        resolveFetch = resolve;
      }),
    );
    render(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />);
    expect(screen.getByText("table-loading")).toBeInTheDocument();

    resolveFetch({ skills: [], count: 0, next_offset: null });
    expect(await screen.findByText("table-loaded")).toBeInTheDocument();
    const expectedQuery = {
      limit: 50,
      offset: 0,
      query: "",
      tag: "",
      security_scan_passed: undefined,
      sort_by: "name",
      sort_order: "asc",
    };
    expect(mockListSkillHubSkills).toHaveBeenCalledWith("sk-test", expectedQuery);
  });
});

describe("ClaudeCodePluginsPanel delete confirmation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
    mockListSkillHubSkills.mockResolvedValue({ skills: [skill], count: 1, next_offset: null });
  });

  it("should ask for confirmation before deleting and name the skill", async () => {
    const user = userEvent.setup();
    render(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />);

    await user.click(await screen.findByTestId("row-delete-plugin-1"));

    expect(await screen.findByText(/are you sure you want to delete skill/i)).toBeInTheDocument();
    expect(screen.getByText("my-skill")).toBeInTheDocument();
    expect(screen.getByText("This action cannot be undone.")).toBeInTheDocument();
    expect(mockDeleteSkillHubSkill).not.toHaveBeenCalled();
  });

  it("should delete the skill and refresh the list once confirmed", async () => {
    const user = userEvent.setup();
    mockDeleteSkillHubSkill.mockResolvedValue({});
    render(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />);

    await user.click(await screen.findByTestId("row-delete-plugin-1"));
    await screen.findByText(/are you sure you want to delete skill/i);
    await user.click(screen.getByRole("button", { name: "Delete" }));

    await waitFor(() => expect(mockDeleteSkillHubSkill).toHaveBeenCalledWith("sk-test", "my-skill"));
    await waitFor(() => expect(mockListSkillHubSkills).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByText(/are you sure you want to delete skill/i)).not.toBeInTheDocument());
  });

  it("should not delete the skill when the confirmation is cancelled", async () => {
    const user = userEvent.setup();
    render(<ClaudeCodePluginsPanel accessToken="sk-test" userRole="Admin" />);

    await user.click(await screen.findByTestId("row-delete-plugin-1"));
    await screen.findByText(/are you sure you want to delete skill/i);
    await user.click(screen.getByRole("button", { name: "Cancel" }));

    await waitFor(() => expect(screen.queryByText(/are you sure you want to delete skill/i)).not.toBeInTheDocument());
    expect(mockDeleteSkillHubSkill).not.toHaveBeenCalled();
  });
});

describe("ClaudeCodePluginsPanel detail access", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    testQueryClient.clear();
    mockListSkillHubSkills.mockResolvedValue({ skills: [skill], count: 1, next_offset: null });
    vi.mocked(getSkillHubSkill).mockResolvedValue(skill);
  });

  it.each([
    ["Admin", true],
    ["proxy_admin", true],
    ["org_admin", false],
    ["Admin Viewer", false],
    ["proxy_admin_viewer", false],
    ["Internal User", false],
    ["internal_user", false],
  ] as const)("limits selected skill download access for %s", async (userRole, canViewDownloads) => {
    render(<ClaudeCodePluginsPanel accessToken="sk-test" userRole={userRole} />);
    fireEvent.doubleClick(await screen.findByTestId(`row-delete-${skill.id}`));
    expect(screen.getByRole("region", { name: `Details for ${skill.name}` })).toBeInTheDocument();
    const downloads = screen.queryByRole("button", { name: "Downloads" });
    expect(downloads !== null).toBe(canViewDownloads);
  });
});
