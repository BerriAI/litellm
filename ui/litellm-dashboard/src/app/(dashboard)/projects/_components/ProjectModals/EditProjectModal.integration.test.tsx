import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { fireEvent, renderWithProviders, screen, waitFor } from "../../../../../../tests/test-utils";
import { EditProjectModal } from "./EditProjectModal";
import { ProjectResponse } from "@/app/(dashboard)/hooks/projects/useProjects";

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

vi.mock("@/lib/toast", () => ({
  toast: { success: vi.fn(), fromError: vi.fn(), error: vi.fn() },
}));

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
    vi.clearAllMocks();
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
