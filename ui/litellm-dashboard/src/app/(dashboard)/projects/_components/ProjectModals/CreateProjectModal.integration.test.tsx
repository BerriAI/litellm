import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { renderWithProviders, screen, waitFor } from "../../../../../../tests/test-utils";
import { CreateProjectModal } from "./CreateProjectModal";

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

vi.mock("@/lib/toast", () => ({
  toast: { success: vi.fn(), fromError: vi.fn(), error: vi.fn() },
}));

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
    vi.clearAllMocks();
  });

  it("sends exactly the antd payload for a minimal project", async () => {
    const user = setup();
    renderModal();

    await user.type(screen.getByLabelText("Project Name"), "My Project");
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

    await user.type(screen.getByLabelText("Project Name"), "My Project");
    await submit(user);

    expect(await screen.findByText("Please select a team")).toBeInTheDocument();
    expect(mutate).not.toHaveBeenCalled();
  });

  it("sends description and max_budget rounded to two decimals", async () => {
    const user = setup();
    renderModal();

    await user.type(screen.getByLabelText("Project Name"), "My Project");
    await pickTeam(user);
    await user.type(screen.getByLabelText("Description"), "Handles support");
    await user.type(screen.getByPlaceholderText("0.00"), "42.567");
    await submit(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(params().description).toBe("Handles support");
    expect(params().max_budget).toBe(42.57);
  });

  it("sends the selected models", async () => {
    const user = setup();
    renderModal();

    await user.type(screen.getByLabelText("Project Name"), "My Project");
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

    await user.type(screen.getByLabelText("Project Name"), "My Project");
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

    await user.type(screen.getByLabelText("Project Name"), "My Project");
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

    await user.type(screen.getByLabelText("Project Name"), "My Project");
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

    await user.type(screen.getByLabelText("Project Name"), "My Project");
    await pickTeam(user);
    await expandAdvanced(user);

    await user.click(screen.getByRole("button", { name: /add model limit/i }));
    await user.type(screen.getByPlaceholderText("Model name (e.g. gpt-4)"), "gpt-4");
    await user.type(screen.getByPlaceholderText("TPM Limit"), "100");
    await user.type(screen.getByPlaceholderText("RPM Limit"), "20");
    await submit(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(params().model_tpm_limit).toStrictEqual({ "gpt-4": 100 });
    expect(params().model_rpm_limit).toStrictEqual({ "gpt-4": 20 });
  });

  it("sends metadata pairs as an object", async () => {
    const user = setup();
    renderModal();

    await user.type(screen.getByLabelText("Project Name"), "My Project");
    await pickTeam(user);
    await expandAdvanced(user);

    await user.click(screen.getByRole("button", { name: /add key-value pair/i }));
    await user.type(screen.getByPlaceholderText("Key"), "owner");
    await user.type(screen.getByPlaceholderText("Value"), "platform");
    await submit(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(params().metadata).toStrictEqual({ owner: "platform" });
  });

  it("sends guardrails when chosen in the advanced section", async () => {
    const user = setup();
    renderModal();

    await user.type(screen.getByLabelText("Project Name"), "My Project");
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

    await user.type(screen.getByLabelText("Project Name"), "My Project");
    await pickTeam(user);
    await expandAdvanced(user);

    await user.click(screen.getByRole("button", { name: /add model limit/i }));
    await user.click(screen.getByRole("button", { name: /add model limit/i }));
    const modelInputs = screen.getAllByPlaceholderText("Model name (e.g. gpt-4)");
    await user.type(modelInputs[0], "gpt-4");
    await user.type(modelInputs[1], "gpt-4");
    await submit(user);

    expect(await screen.findByText("Duplicate model")).toBeInTheDocument();
    expect(mutate).not.toHaveBeenCalled();
  });
});
