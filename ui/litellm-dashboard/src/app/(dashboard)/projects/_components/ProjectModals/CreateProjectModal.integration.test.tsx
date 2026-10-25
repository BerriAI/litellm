import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { act, fireEvent, renderWithProviders, screen, waitFor } from "../../../../../../tests/test-utils";
import { CreateProjectModal } from "./CreateProjectModal";
import LanguageSelector from "@/components/LanguageSelector";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { translate, type Language } from "@/i18n/language";
import { fetchTeamModels } from "@/components/organisms/create_key_button";
import { getGuardrailsList } from "@/components/networking";
import { Toaster } from "@/components/ui/sonner";
import { toast } from "@/lib/toast";

const mutate = vi.fn();

vi.mock("@/app/(dashboard)/hooks/projects/useCreateProject", () => ({
  useCreateProject: () => ({ mutate, isPending: false }),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "test-token", userId: "u-1", userRole: "Admin" }),
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useTeams: () => ({
    data: [
      { team_id: "team-1", team_alias: "Engineering", models: ["gpt-4"] },
      { team_id: "team-2", team_alias: "Sales", models: [] },
    ],
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

const renderModal = () => renderWithProviders(<CreateProjectModal isOpen onClose={vi.fn()} />);

const pickTeam = async (user: User) => {
  await user.click(screen.getByLabelText("Team"));
  await user.click(await screen.findByText("Engineering"));
};

const submit = async (user: User) => user.click(screen.getByRole("button", { name: /create project/i }));

const expandAdvanced = async (user: User) => {
  await user.click(screen.getByText("Advanced Settings"));
  await screen.findByText("Model-Specific Limits");
};

const params = () => mutate.mock.calls.at(-1)?.[0] as Record<string, unknown>;

describe("CreateProjectModal submit payload", () => {
  beforeEach(() => {
    localStorage.clear();
    toast.dismiss();
    vi.clearAllMocks();
  });

  afterEach(() => act(() => toast.dismiss()));

  it.each(["success", "failure", "server", "envelope"])(
    "项目创建的 %s 通知跟随当前语言且保留服务端消息",
    async (result) => {
      const user = setup();
      renderWithProviders(
        <LanguageProvider defaultLanguage="en" userId="project-create-notice" storageScope="project-create-notice">
          <LanguageSelector />
          <CreateProjectModal isOpen onClose={vi.fn()} />
          <Toaster expand />
        </LanguageProvider>,
      );
      fireEvent.change(screen.getByLabelText("Project Name"), { target: { value: "Settings" } });
      await pickTeam(user);
      await submit(user);
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
      const message = result === "success" ? "Project created successfully" : "Failed to create project";
      expect(await screen.findByText(serverMessage ? "Home" : translate("zh-CN", message))).toBeVisible();
      expect(screen.queryByText("首页")).not.toBeInTheDocument();
      expect(screen.queryByText(/internal implementation detail/)).not.toBeInTheDocument();
      fireEvent.change(screen.getByLabelText("界面语言"), { target: { value: "en" } });
      expect(await screen.findByText(serverMessage ? "Home" : message)).toBeVisible();
      expect(mutate).toHaveBeenCalledTimes(1);
      expect(params()).toMatchObject({ project_alias: "Settings", team_id: "team-1" });
      if (result !== "success") expect(screen.getByLabelText("Project Name")).toHaveValue("Settings");
    },
  );

  it.each<Language>(["en", "zh-CN"])("项目校验随 %s 切换且保留嵌套草稿和提交值", async (initialLanguage) => {
    const user = setup();
    const nextLanguage = initialLanguage === "en" ? "zh-CN" : "en";
    renderWithProviders(
      <LanguageProvider defaultLanguage={initialLanguage} userId="project-create" storageScope="project-create">
        <LanguageSelector />
        <CreateProjectModal isOpen onClose={vi.fn()} />
      </LanguageProvider>,
    );
    fireEvent.change(screen.getByLabelText(translate(initialLanguage, "Description")), {
      target: { value: "Settings" },
    });
    await user.click(screen.getByText(translate(initialLanguage, "Advanced Settings")));
    await user.click(screen.getByRole("button", { name: translate(initialLanguage, "Add Model Limit") }));
    await user.click(screen.getByRole("button", { name: translate(initialLanguage, "Add Key-Value Pair") }));
    await user.click(screen.getByRole("button", { name: translate(initialLanguage, "Create Project") }));
    const messages = [
      "Please enter a project name",
      "Please select a team",
      "Missing model",
      "Missing key",
      "Missing value",
    ];
    for (const message of messages) {
      expect(await screen.findByText(translate(initialLanguage, message))).toBeVisible();
    }
    await waitFor(() => expect(getGuardrailsList).toHaveBeenCalledTimes(1));

    // 仅切换语言，不重提表单或重新获取资源来刷新校验文案。
    fireEvent.change(screen.getByLabelText(translate(initialLanguage, "Interface language")), {
      target: { value: nextLanguage },
    });
    for (const message of messages) {
      expect(screen.getByText(translate(nextLanguage, message))).toBeVisible();
      expect(screen.queryByText(translate(initialLanguage, message))).not.toBeInTheDocument();
    }
    expect(screen.getByLabelText(translate(nextLanguage, "Description"))).toHaveValue("Settings");
    expect(getGuardrailsList).toHaveBeenCalledTimes(1);
    expect(fetchTeamModels).not.toHaveBeenCalled();
    expect(mutate).not.toHaveBeenCalled();

    fireEvent.change(screen.getByLabelText(translate(nextLanguage, "Project Name")), { target: { value: "Home" } });
    await user.click(screen.getByLabelText(translate(nextLanguage, "Team")));
    await user.click(await screen.findByText("Engineering"));
    fireEvent.change(screen.getByPlaceholderText(translate(nextLanguage, "Model name (e.g. gpt-4)")), {
      target: { value: "Home" },
    });
    fireEvent.change(screen.getByLabelText(translate(nextLanguage, "RPM Limit")), { target: { value: "5" } });
    fireEvent.change(screen.getByPlaceholderText(translate(nextLanguage, "Metadata key")), {
      target: { value: "Home" },
    });
    fireEvent.change(screen.getByPlaceholderText(translate(nextLanguage, "Value")), { target: { value: "Settings" } });
    await user.click(screen.getByRole("button", { name: translate(nextLanguage, "Create Project") }));
    await waitFor(() => expect(mutate).toHaveBeenCalledTimes(1));
    const expected = {
      project_alias: "Home",
      description: "Settings",
      team_id: "team-1",
      model_rpm_limit: { Home: 5 },
      metadata: { Home: "Settings" },
    };
    expect(params()).toMatchObject(expected);
  });

  it("模型下拉提示翻译且切换语言不丢失团队与模型选择", async () => {
    const user = setup();
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId="project-models" storageScope="project-models">
        <LanguageSelector />
        <CreateProjectModal isOpen onClose={vi.fn()} />
      </LanguageProvider>,
    );
    expect(
      screen.getByLabelText(translate("zh-CN", "Allowed Models (scoped to selected team's models)")),
    ).toHaveTextContent("请先选择团队");
    fireEvent.change(screen.getByLabelText(translate("zh-CN", "Interface language")), { target: { value: "en" } });
    expect(screen.getByLabelText("Allowed Models (scoped to selected team's models)")).toHaveTextContent(
      "Select a team first",
    );
    fireEvent.change(screen.getByLabelText("Project Name"), { target: { value: "Home" } });
    await pickTeam(user);
    expect(screen.getByLabelText("Allowed Models (scoped to selected team's models)")).toHaveTextContent(
      "Select models",
    );
    await waitFor(() => expect(fetchTeamModels).toHaveBeenCalledTimes(1));
    fireEvent.change(screen.getByLabelText("Interface language"), { target: { value: "zh-CN" } });
    const models = screen.getByLabelText(translate("zh-CN", "Allowed Models (scoped to selected team's models)"));
    expect(models).toHaveTextContent(translate("zh-CN", "Select models"));
    await user.click(models);
    await user.click(await screen.findByTitle("gpt-4"));
    fireEvent.change(screen.getByLabelText(translate("zh-CN", "Interface language")), { target: { value: "en" } });
    expect(screen.getByLabelText("Allowed Models (scoped to selected team's models)")).toHaveTextContent("gpt-4");
    expect(screen.getByLabelText("Team")).toHaveValue("Engineering");
    expect(fetchTeamModels).toHaveBeenCalledTimes(1);
    expect(getGuardrailsList).toHaveBeenCalledTimes(1);
    await submit(user);
    await waitFor(() => expect(mutate).toHaveBeenCalledTimes(1));
    expect(params()).toMatchObject({ project_alias: "Home", team_id: "team-1", models: ["gpt-4"] });
  });

  it("sends exactly the antd payload for a minimal project", async () => {
    const user = setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Project Name"), { target: { value: "My Project" } });
    await pickTeam(user);
    await submit(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());

    expect(mutate).toHaveBeenCalledTimes(1);
    expect(params()).toStrictEqual({
      project_alias: "My Project",
      description: undefined,
      models: [],
      max_budget: undefined,
      blocked: false,
      team_id: "team-1",
    });
  });

  it("does not submit without a project name", async () => {
    const user = setup();
    renderModal();

    await pickTeam(user);
    await submit(user);

    expect(await screen.findByText("Please enter a project name")).toBeInTheDocument();
    expect(mutate).not.toHaveBeenCalled();
  });

  it("does not submit without a team", async () => {
    const user = setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Project Name"), { target: { value: "My Project" } });
    await submit(user);

    expect(await screen.findByText("Please select a team")).toBeInTheDocument();
    expect(mutate).not.toHaveBeenCalled();
  });

  it("sends description and max_budget rounded to two decimals", async () => {
    const user = setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Project Name"), { target: { value: "My Project" } });
    await pickTeam(user);
    fireEvent.change(screen.getByLabelText("Description"), { target: { value: "Handles support" } });
    fireEvent.change(screen.getByPlaceholderText("0.00"), { target: { value: "42.567" } });
    await submit(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(params().description).toBe("Handles support");
    expect(params().max_budget).toBe(42.57);
  });

  it("sends the selected models", async () => {
    const user = setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Project Name"), { target: { value: "My Project" } });
    await pickTeam(user);
    await user.click(screen.getByLabelText(/Allowed Models/));
    await user.click(await screen.findByTitle("gpt-4"));
    await submit(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(params().models).toStrictEqual(["gpt-4"]);
  });

  it("collapses models to all-team-models when that option is chosen", async () => {
    const user = setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Project Name"), { target: { value: "My Project" } });
    await pickTeam(user);
    await user.click(screen.getByLabelText(/Allowed Models/));
    await user.click(await screen.findByTitle("gpt-4"));
    await user.click(await screen.findByTitle("All Team Models"));
    await submit(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(params().models).toStrictEqual(["all-team-models"]);
  });

  it("sends blocked true when the Block Project switch is on", async () => {
    const user = setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Project Name"), { target: { value: "My Project" } });
    await pickTeam(user);
    await expandAdvanced(user);
    await user.click(screen.getByRole("switch"));
    await submit(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(params().blocked).toBe(true);
  });

  it("omits guardrails, limits and metadata when the advanced section is untouched", async () => {
    const user = setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Project Name"), { target: { value: "My Project" } });
    await pickTeam(user);
    await expandAdvanced(user);
    await submit(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(params()).not.toHaveProperty("guardrails");
    expect(params()).not.toHaveProperty("model_rpm_limit");
    expect(params()).not.toHaveProperty("model_tpm_limit");
    expect(params()).not.toHaveProperty("metadata");
  });

  it("sends model limits as rpm and tpm maps keyed by model", async () => {
    const user = setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Project Name"), { target: { value: "My Project" } });
    await pickTeam(user);
    await expandAdvanced(user);

    await user.click(screen.getByRole("button", { name: /add model limit/i }));
    fireEvent.change(screen.getByPlaceholderText("Model name (e.g. gpt-4)"), { target: { value: "gpt-4" } });
    fireEvent.change(screen.getByPlaceholderText("TPM Limit"), { target: { value: "100" } });
    fireEvent.change(screen.getByPlaceholderText("RPM Limit"), { target: { value: "20" } });
    fireEvent.change(screen.getByPlaceholderText("Input TPM Limit"), { target: { value: "60" } });
    fireEvent.change(screen.getByPlaceholderText("Output TPM Limit"), { target: { value: "40" } });
    await submit(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(params().model_tpm_limit).toStrictEqual({ "gpt-4": 100 });
    expect(params().model_rpm_limit).toStrictEqual({ "gpt-4": 20 });
    expect(params().model_itpm_limit).toStrictEqual({ "gpt-4": 60 });
    expect(params().model_otpm_limit).toStrictEqual({ "gpt-4": 40 });
  });

  it("sends metadata pairs as an object", async () => {
    const user = setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Project Name"), { target: { value: "My Project" } });
    await pickTeam(user);
    await expandAdvanced(user);

    await user.click(screen.getByRole("button", { name: /add key-value pair/i }));
    fireEvent.change(screen.getByPlaceholderText("Metadata key"), { target: { value: "owner" } });
    fireEvent.change(screen.getByPlaceholderText("Value"), { target: { value: "platform" } });
    await submit(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(params().metadata).toStrictEqual({ owner: "platform" });
  });

  it("sends guardrails when chosen in the advanced section", async () => {
    const user = setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Project Name"), { target: { value: "My Project" } });
    await pickTeam(user);
    await expandAdvanced(user);

    await user.click(screen.getByLabelText("Guardrails"));
    await user.click(await screen.findByTitle("pii-guard"));
    await user.keyboard("{Escape}");
    await submit(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(params().guardrails).toStrictEqual(["pii-guard"]);
  });

  it("blocks submit when two model limits name the same model", async () => {
    const user = setup();
    renderModal();

    fireEvent.change(screen.getByLabelText("Project Name"), { target: { value: "My Project" } });
    await pickTeam(user);
    await expandAdvanced(user);

    await user.click(screen.getByRole("button", { name: /add model limit/i }));
    await user.click(screen.getByRole("button", { name: /add model limit/i }));
    const modelInputs = screen.getAllByPlaceholderText("Model name (e.g. gpt-4)");
    fireEvent.change(modelInputs[0], { target: { value: "gpt-4" } });
    fireEvent.change(modelInputs[1], { target: { value: "gpt-4" } });
    await submit(user);

    expect(await screen.findByText("Duplicate model")).toBeInTheDocument();
    expect(mutate).not.toHaveBeenCalled();
  });
});
