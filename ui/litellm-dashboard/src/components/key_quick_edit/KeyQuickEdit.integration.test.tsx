import { beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { fireEvent, waitFor } from "@testing-library/react";
import { renderWithProviders, screen, testQueryClient } from "../../../tests/test-utils";
import type { KeyResponse, Team } from "@/components/key_team_helpers/key_list";
import { isTeamAdminEditingMemberKey } from "@/components/templates/teamAdminMemberKeyPayload";
import { canModifyKey } from "./canModifyKey";
import { KeyBudgetQuickEdit } from "./KeyBudgetQuickEdit";
import { KeyModelsQuickEdit } from "./KeyModelsQuickEdit";
import { keyUpdateCall, modelAvailableCall } from "../networking";

vi.mock("../networking", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../networking")>()),
  keyUpdateCall: vi.fn(),
  modelAvailableCall: vi.fn(),
}));

const keyData = (overrides: Partial<KeyResponse> = {}): KeyResponse =>
  ({
    token: "key-token",
    token_id: "key-token-id",
    key_name: "quick-edit-key",
    key_alias: "quick-edit-key",
    user_id: "owner",
    team_id: null,
    models: ["gpt-4o-mini"],
    max_budget: 25,
    budget_duration: "30d",
    spend: 3.25,
    allowed_routes: [],
    key_type: null,
    ...overrides,
  }) as KeyResponse;

const makeTeam = (): Team =>
  ({
    team_id: "team-1",
    models: ["gpt-4o-mini"],
    members_with_roles: [{ user_id: "team-admin", role: "admin" }],
  }) as Team;

const QuickEditControls = ({
  keyData,
  userId,
  userRole,
  teams,
}: {
  keyData: KeyResponse;
  userId: string;
  userRole: string;
  teams: Team[];
}) => {
  const permissionContext = { userRole, userId, key: keyData, teams };
  const canModify = canModifyKey(permissionContext);
  const team = teams.find((candidate) => candidate.team_id === keyData.team_id);
  const memberKeyContext = {
    userRole,
    userId,
    keyUserId: keyData.user_id,
    keyTeamId: keyData.team_id,
    teamMembers: team?.members_with_roles,
  };
  const isTeamAdminEditingMember = isTeamAdminEditingMemberKey(memberKeyContext);

  return (
    <div className="group/editable">
      <KeyBudgetQuickEdit keyData={keyData} accessToken="access-token" canModify={canModify} />
      <KeyModelsQuickEdit
        keyData={keyData}
        team={team}
        accessToken="access-token"
        userId={userId}
        userRole={userRole}
        canModify={canModify}
        canEditModels={!isTeamAdminEditingMember}
      />
    </div>
  );
};

beforeEach(() => {
  testQueryClient.clear();
  vi.clearAllMocks();
  vi.mocked(keyUpdateCall).mockResolvedValue({});
  vi.mocked(modelAvailableCall).mockResolvedValue({
    data: [{ id: "gpt-4o-mini" }, { id: "gpt-3.5-turbo" }],
  });
});

describe("key quick edit popovers", () => {
  it("saves a budget with Enter using only the budget fields", async () => {
    const user = userEvent.setup();
    renderWithProviders(<QuickEditControls keyData={keyData()} userId="owner" userRole="Admin" teams={[]} />);

    await user.click(screen.getByRole("button", { name: "Edit budget" }));
    const budgetInput = screen.getByRole("spinbutton", { name: "Max budget (USD)" });
    await user.clear(budgetInput);
    await user.type(budgetInput, "50{Enter}");

    await waitFor(() => expect(keyUpdateCall).toHaveBeenCalledTimes(1));
    expect(screen.queryByText("Quick edit budget")).not.toBeInTheDocument();
    expect(keyUpdateCall).toHaveBeenCalledTimes(1);
    expect(keyUpdateCall).toHaveBeenCalledWith("access-token", {
      key: "key-token",
      max_budget: 50,
      budget_duration: "30d",
    });
  });

  it("clears the budget by sending null", async () => {
    const user = userEvent.setup();
    renderWithProviders(<QuickEditControls keyData={keyData()} userId="owner" userRole="Admin" teams={[]} />);

    await user.click(screen.getByRole("button", { name: "Edit budget" }));
    fireEvent.change(screen.getByRole("spinbutton", { name: "Max budget (USD)" }), { target: { value: "" } });
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(keyUpdateCall).toHaveBeenCalledTimes(1);
    expect(keyUpdateCall).toHaveBeenCalledWith("access-token", {
      key: "key-token",
      max_budget: null,
      budget_duration: "30d",
    });
  });

  it("saves model selections with only the models field", async () => {
    const user = userEvent.setup();
    renderWithProviders(<QuickEditControls keyData={keyData()} userId="owner" userRole="Admin" teams={[]} />);

    expect(modelAvailableCall).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Edit models" }));
    const modelsInput = await screen.findByRole("combobox", { name: "Select models" });
    await waitFor(() => expect(modelsInput).toBeEnabled());
    await user.click(modelsInput);
    await screen.findByRole("option", { name: "gpt-4o-mini" });
    await user.type(modelsInput, "gpt-3.5");
    await user.click(await screen.findByRole("option", { name: "gpt-3.5-turbo" }));
    await user.click(screen.getByRole("button", { name: "Save" }));

    expect(keyUpdateCall).toHaveBeenCalledTimes(1);
    expect(keyUpdateCall).toHaveBeenCalledWith("access-token", {
      key: "key-token",
      models: ["gpt-4o-mini", "gpt-3.5-turbo"],
    });
  });

  it("allows team admins to edit budgets but hides models for a member key", () => {
    renderWithProviders(
      <QuickEditControls
        keyData={keyData({ team_id: "team-1", user_id: "team-member" })}
        userId="team-admin"
        userRole="Internal User"
        teams={[makeTeam()]}
      />,
    );

    expect(screen.getByRole("button", { name: "Edit budget" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit models" })).not.toBeInTheDocument();
  });

  it("hides both pencils from internal viewers, including key owners", () => {
    renderWithProviders(<QuickEditControls keyData={keyData()} userId="owner" userRole="Internal Viewer" teams={[]} />);

    expect(screen.queryByRole("button", { name: "Edit budget" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit models" })).not.toBeInTheDocument();
  });
});
