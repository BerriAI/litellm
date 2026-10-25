import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { act, fireEvent, renderWithProviders, screen, waitFor } from "../../../../../../tests/test-utils";
import { EditProjectModal } from "./EditProjectModal";
import { ProjectResponse } from "@/app/(dashboard)/hooks/projects/useProjects";
import LanguageSelector from "@/components/LanguageSelector";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { translate, type Language } from "@/i18n/language";
import { fetchTeamModels } from "@/components/organisms/create_key_button";
import { getGuardrailsList } from "@/components/networking";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "@/lib/toast";

const mutate = vi.fn();

vi.mock("@/app/(dashboard)/hooks/projects/useUpdateProject", () => ({
  useUpdateProject: () => ({ mutate, isPending: false }),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "test-token", userId: "u-1", userRole: "Admin" }),
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeams: () => ({
    data: [{ team_id: "team-1", team_alias: "Engineering", models: ["gpt-4"] }],
    isLoading: false,
  }),
}));

vi.mock("@/components/organisms/create_key_button", () => ({
  fetchTeamModels: vi.fn().mockResolvedValue([]),
}));

vi.mock("@/components/networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/components/networking")>()),
  getGuardrailsList: vi.fn().mockResolvedValue({ guardrails: [{ guardrail_name: "pii-guard" }] }),
}));

vi.mock("@/components/key_team_helpers/fetch_available_models_team_key", () => ({
  getModelDisplayName: (model: string) => model,
}));

vi.unmock("@/lib/toast");

const setup = () => userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
type User = ReturnType<typeof setup>;

const project: ProjectResponse = {
  project_id: "proj-1",
  project_alias: "My Project",
  description: "A test project",
  team_id: "team-1",
  budget_id: null,
  metadata: {
    owner: "platform",
    guardrails: ["pii-guard"],
    model_rpm_limit: { "gpt-4": 20 },
    model_tpm_limit: { "gpt-4": 100 },
    model_itpm_limit: { "gpt-4": 60 },
    model_otpm_limit: { "gpt-4": 40 },
  },
  models: ["gpt-4"],
  spend: 10,
  model_spend: null,
  model_rpm_limit: null,
  model_tpm_limit: null,
  blocked: false,
  object_permission_id: null,
  created_at: "2024-01-01T00:00:00Z",
  created_by: "user-1",
  updated_at: "2024-01-02T00:00:00Z",
  updated_by: "user-1",
  litellm_budget_table: { max_budget: 50 },
} as unknown as ProjectResponse;

const renderModal = (data: ProjectResponse = project) =>
  renderWithProviders(<EditProjectModal isOpen project={data} onClose={vi.fn()} />);

const save = async (user: User) => user.click(screen.getByRole("button", { name: /save changes/i }));

const variables = () => mutate.mock.calls.at(-1)?.[0] as { projectId: string; params: Record<string, unknown> };

describe("EditProjectModal submit payload", () => {
  beforeEach(() => {
    localStorage.clear();
    toast.dismiss();
    vi.clearAllMocks();
  });

  afterEach(() => act(() => toast.dismiss()));

  it.each(["success", "failure", "server", "envelope"])(
    "项目保存的 %s 通知跟随当前语言且保留服务端消息",
    async (result) => {
      const user = setup();
      renderWithProviders(
        <LanguageProvider defaultLanguage="en" userId="project-edit-notice" storageScope="project-edit-notice">
          <LanguageSelector />
          <EditProjectModal isOpen project={project} onClose={vi.fn()} />
          <Toaster expand />
        </LanguageProvider>,
      );
      fireEvent.change(screen.getByLabelText("Project Name"), { target: { value: "Settings" } });
      await save(user);
      await waitFor(() => expect(mutate).toHaveBeenCalledTimes(1));
      const callbacks = mutate.mock.calls[0][1];
      const serverMessage = result === "server" || result === "envelope";
      const body = { error: { message: "Home", internal: "internal implementation detail" } };
      const serverError = result === "envelope" ? JSON.stringify(body) : "Home";
      fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
      act(() => {
        if (result === "success") callbacks.onSuccess();
        else callbacks.onError(new Error(serverMessage ? serverError : ""));
      });
      const message = result === "success" ? "Project updated successfully" : "Failed to update project";
      expect(await screen.findByText(serverMessage ? "Home" : translate("zh-CN", message))).toBeVisible();
      expect(screen.queryByText("首页")).not.toBeInTheDocument();
      expect(screen.queryByText(/internal implementation detail/)).not.toBeInTheDocument();
      fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
      expect(await screen.findByText(serverMessage ? "Home" : message)).toBeVisible();
      expect(mutate).toHaveBeenCalledTimes(1);
      expect(variables()).toMatchObject({
        projectId: "proj-1",
        params: { project_alias: "Settings", team_id: "team-1" },
      });
      if (result !== "success") expect(screen.getByLabelText("Project Name")).toHaveValue("Settings");
    },
  );

  it.each<Language>(["en", "zh-CN"])("重复字段校验随 %s 切换且保留编辑值与项目标识", async (initialLanguage) => {
    const user = setup();
    const nextLanguage = initialLanguage === "en" ? "zh-CN" : "en";
    renderWithProviders(
      <LanguageProvider defaultLanguage={initialLanguage} userId="project-edit" storageScope="project-edit">
        <LanguageSelector />
        <EditProjectModal isOpen project={project} onClose={vi.fn()} />
      </LanguageProvider>,
    );
    fireEvent.change(screen.getByLabelText(translate(initialLanguage, "Project Name")), { target: { value: "Home" } });
    await user.click(screen.getByText(translate(initialLanguage, "Advanced Settings")));
    await user.click(screen.getByRole("button", { name: translate(initialLanguage, "Add Model Limit") }));
    fireEvent.change(screen.getAllByLabelText(translate(initialLanguage, "Model"), { exact: true })[1], {
      target: { value: "gpt-4" },
    });
    fireEvent.change(screen.getAllByLabelText(translate(initialLanguage, "RPM Limit"), { exact: true })[1], {
      target: { value: "5" },
    });
    await user.click(screen.getByRole("button", { name: translate(initialLanguage, "Add Key-Value Pair") }));
    fireEvent.change(screen.getAllByPlaceholderText(translate(initialLanguage, "Metadata key"))[1], {
      target: { value: "owner" },
    });
    fireEvent.change(screen.getAllByPlaceholderText(translate(initialLanguage, "Value"))[1], {
      target: { value: "Settings" },
    });
    await user.click(screen.getByRole("button", { name: translate(initialLanguage, "Save Changes") }));
    expect(await screen.findByText(translate(initialLanguage, "Duplicate model"))).toBeVisible();
    expect(screen.getByText(translate(initialLanguage, "Duplicate key"))).toBeVisible();
    await waitFor(() => expect(fetchTeamModels).toHaveBeenCalledTimes(1));

    // 已有错误更新语言时，不重建表单，也不发送保存或资源查询。
    fireEvent.change(screen.getByLabelText(translate(initialLanguage, "Interface language")), {
      target: { value: nextLanguage },
    });
    expect(screen.getByText(translate(nextLanguage, "Duplicate model"))).toBeVisible();
    expect(screen.getByText(translate(nextLanguage, "Duplicate key"))).toBeVisible();
    expect(screen.queryByText(translate(initialLanguage, "Duplicate model"))).not.toBeInTheDocument();
    expect(screen.getByLabelText(translate(nextLanguage, "Project Name"))).toHaveValue("Home");
    expect(screen.getAllByPlaceholderText(translate(nextLanguage, "Value"))[1]).toHaveValue("Settings");
    expect(fetchTeamModels).toHaveBeenCalledTimes(1);
    expect(getGuardrailsList).toHaveBeenCalledTimes(1);
    expect(mutate).not.toHaveBeenCalled();
    fireEvent.change(screen.getAllByLabelText(translate(nextLanguage, "Model"), { exact: true })[1], {
      target: { value: "Settings" },
    });
    fireEvent.change(screen.getAllByPlaceholderText(translate(nextLanguage, "Metadata key"))[1], {
      target: { value: "Home" },
    });
    await user.click(screen.getByRole("button", { name: translate(nextLanguage, "Save Changes") }));
    await waitFor(() => expect(mutate).toHaveBeenCalledTimes(1));
    expect(variables().projectId).toBe("proj-1");
    const expected = {
      project_alias: "Home",
      team_id: "team-1",
      models: ["gpt-4"],
      model_rpm_limit: { "gpt-4": 20, Settings: 5 },
      metadata: { owner: "platform", Home: "Settings" },
    };
    expect(variables().params).toMatchObject(expected);
  });

  it("sends exactly the antd payload for an untouched save", async () => {
    const user = setup();
    renderModal();
    await screen.findByDisplayValue("My Project");

    await save(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());

    expect(mutate).toHaveBeenCalledTimes(1);
    expect(variables().projectId).toBe("proj-1");
    expect(variables().params).toStrictEqual({
      project_alias: "My Project",
      description: "A test project",
      models: ["gpt-4"],
      max_budget: 50,
      blocked: false,
      team_id: "team-1",
    });
  });

  it("should send an explicit clear after blanking a saved budget", async () => {
    const user = setup();
    renderModal();

    const budgetInput = screen.getByRole("spinbutton", { name: "Max Budget (USD)" });
    await user.clear(budgetInput);
    await user.tab();
    expect(budgetInput).toHaveValue(null);
    await save(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(JSON.parse(JSON.stringify(variables().params))).toMatchObject({ max_budget: null });
  });

  it("includes the advanced fields once Advanced Settings has been opened, even after collapsing it again", async () => {
    const user = setup();
    renderModal();
    await screen.findByDisplayValue("My Project");

    await user.click(screen.getByText("Advanced Settings"));
    await screen.findByText("Model-Specific Limits");
    await user.click(screen.getByText("Advanced Settings"));
    await save(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(variables().params).toStrictEqual({
      project_alias: "My Project",
      description: "A test project",
      models: ["gpt-4"],
      max_budget: 50,
      blocked: false,
      guardrails: ["pii-guard"],
      model_rpm_limit: { "gpt-4": 20 },
      model_tpm_limit: { "gpt-4": 100 },
      model_itpm_limit: { "gpt-4": 60 },
      model_otpm_limit: { "gpt-4": 40 },
      metadata: { owner: "platform" },
      team_id: "team-1",
    });
  });

  it("never sends server-only fields from the loaded record", async () => {
    const user = setup();
    renderModal();
    await screen.findByDisplayValue("My Project");

    await save(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(variables().params).not.toHaveProperty("created_at");
    expect(variables().params).not.toHaveProperty("updated_at");
    expect(variables().params).not.toHaveProperty("project_id");
    expect(variables().params).not.toHaveProperty("spend");
    expect(variables().params).not.toHaveProperty("budget_id");
    expect(variables().params).not.toHaveProperty("litellm_budget_table");
  });

  it("keeps the internal metadata keys out of user metadata", async () => {
    const user = setup();
    renderModal();
    await screen.findByDisplayValue("My Project");

    await user.click(screen.getByText("Advanced Settings"));
    await screen.findByText("Model-Specific Limits");
    await save(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(variables().params.metadata).toStrictEqual({ owner: "platform" });
  });

  it("blocks the save when the project name is cleared", async () => {
    const user = setup();
    renderModal();

    await user.clear(screen.getByLabelText("Project Name"));
    await save(user);

    expect(await screen.findByText("Please enter a project name")).toBeInTheDocument();
    expect(mutate).not.toHaveBeenCalled();
  });

  it("sends an edited project name", async () => {
    const user = setup();
    renderModal();
    const nameInput = await screen.findByDisplayValue("My Project");

    await user.clear(nameInput);
    fireEvent.change(nameInput, { target: { value: "Renamed" } });
    await save(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(variables().params.project_alias).toBe("Renamed");
  });

  it("omits max_budget when the project has no budget row", async () => {
    const user = setup();
    renderModal({ ...project, litellm_budget_table: null } as unknown as ProjectResponse);
    await screen.findByDisplayValue("My Project");

    await save(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(variables().params.max_budget).toBeUndefined();
  });

  it("omits guardrails, limits and metadata for a project with empty metadata", async () => {
    const user = setup();
    renderModal({ ...project, metadata: null } as unknown as ProjectResponse);
    await screen.findByDisplayValue("My Project");

    await save(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(variables().params).not.toHaveProperty("guardrails");
    expect(variables().params).not.toHaveProperty("model_rpm_limit");
    expect(variables().params).not.toHaveProperty("model_tpm_limit");
    expect(variables().params).not.toHaveProperty("model_itpm_limit");
    expect(variables().params).not.toHaveProperty("model_otpm_limit");
    expect(variables().params).not.toHaveProperty("metadata");
  });

  it("sends empty limit maps once the model limit row is removed, so the stored limits are cleared", async () => {
    const user = setup();
    renderModal();
    await screen.findByDisplayValue("My Project");

    await user.click(screen.getByText("Advanced Settings"));
    await screen.findByText("Model-Specific Limits");
    await user.click(screen.getByRole("button", { name: "Remove model limit 1" }));
    await save(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(variables().params.model_itpm_limit).toStrictEqual({});
    expect(variables().params.model_otpm_limit).toStrictEqual({});
    expect(variables().params.model_tpm_limit).toStrictEqual({});
    expect(variables().params.model_rpm_limit).toStrictEqual({});
    expect(variables().params.metadata).toStrictEqual({ owner: "platform" });
  });

  it("sends an empty input TPM map when only that field is blanked on a row that keeps its other limits", async () => {
    const user = setup();
    renderModal();
    await screen.findByDisplayValue("My Project");

    await user.click(screen.getByText("Advanced Settings"));
    await screen.findByText("Model-Specific Limits");
    await user.clear(screen.getByLabelText("Input TPM Limit"));
    await save(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(variables().params.model_itpm_limit).toStrictEqual({});
    expect(variables().params.model_otpm_limit).toStrictEqual({ "gpt-4": 40 });
    expect(variables().params.model_tpm_limit).toStrictEqual({ "gpt-4": 100 });
  });

  it("sends an empty metadata object once the last metadata row is removed", async () => {
    const user = setup();
    renderModal({ ...project, metadata: { owner: "platform" } } as unknown as ProjectResponse);
    await screen.findByDisplayValue("My Project");

    await user.click(screen.getByText("Advanced Settings"));
    await screen.findByText("Metadata");
    await user.click(screen.getByRole("button", { name: "Remove metadata pair 1" }));
    await save(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(variables().params.metadata).toStrictEqual({});
  });

  it("round-trips input and output-only model limits from project metadata", async () => {
    const user = setup();
    renderModal({
      ...project,
      metadata: {
        model_itpm_limit: { "input-model": 150 },
        model_otpm_limit: { "output-model": 250 },
      },
    } as unknown as ProjectResponse);
    await screen.findByDisplayValue("My Project");

    await user.click(screen.getByText("Advanced Settings"));
    await screen.findByText("Model-Specific Limits");
    await save(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(variables().params.model_itpm_limit).toStrictEqual({ "input-model": 150 });
    expect(variables().params.model_otpm_limit).toStrictEqual({ "output-model": 250 });
    expect(variables().params.metadata).toStrictEqual({});
  });
});
