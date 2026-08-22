import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { fireEvent, renderWithProviders, screen, waitFor } from "../../../../../../tests/test-utils";
import { AccessGroupEditModal } from "./AccessGroupEditModal";
import { AccessGroupResponse } from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";

const mutate = vi.fn();

vi.mock("@/app/(dashboard)/hooks/accessGroups/useEditAccessGroup", () => ({
  useEditAccessGroup: () => ({ mutate, isPending: false }),
}));

vi.mock("@/app/(dashboard)/hooks/agents/useAgents", () => ({
  useAgents: () => ({ data: { agents: [{ agent_id: "agent-1", agent_name: "Support Bot" }] } }),
}));

vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPServers", () => ({
  useMCPServers: () => ({ data: [{ server_id: "srv-1", server_name: "Files" }] }),
}));

vi.mock("@/components/ModelSelect/ModelSelect", () => ({
  ModelSelect: ({ value, onChange }: { value: string[]; onChange: (next: string[]) => void }) => (
    <button type="button" aria-label="model-select" onClick={() => onChange([...(value ?? []), "gpt-4"])}>
      {(value ?? []).join(",")}
    </button>
  ),
}));

vi.mock("@/lib/toast", () => ({
  toast: { success: vi.fn(), fromError: vi.fn(), error: vi.fn() },
}));

const setup = () => userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
type User = ReturnType<typeof setup>;

const accessGroup: AccessGroupResponse = {
  access_group_id: "ag-1",
  access_group_name: "Engineering",
  description: "Engineers",
  access_model_names: ["gpt-4"],
  access_mcp_server_ids: ["srv-1"],
  access_agent_ids: ["agent-1"],
  assigned_team_ids: [],
  assigned_key_ids: [],
  created_at: "2024-01-01T00:00:00Z",
  created_by: "user-1",
  updated_at: "2024-01-02T00:00:00Z",
  updated_by: "user-1",
};

const renderModal = (data: AccessGroupResponse = accessGroup) =>
  renderWithProviders(<AccessGroupEditModal visible accessGroup={data} onCancel={vi.fn()} />);

const save = async (user: User) => user.click(screen.getByRole("button", { name: "Save Changes" }));

const variables = () => mutate.mock.calls.at(-1)?.[0] as { accessGroupId: string; params: Record<string, unknown> };

describe("AccessGroupEditModal submit payload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("sends exactly the antd payload for an untouched save", async () => {
    const user = setup();
    renderModal();
    await screen.findByDisplayValue("Engineering");

    await save(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());

    expect(mutate).toHaveBeenCalledTimes(1);
    expect(variables().accessGroupId).toBe("ag-1");
    expect(variables().params).toStrictEqual({
      access_group_name: "Engineering",
      description: "Engineers",
      access_model_names: undefined,
      access_mcp_server_ids: undefined,
      access_agent_ids: undefined,
    });
  });

  it("sends a tab's field only once that tab has been visited", async () => {
    const user = setup();
    renderModal();
    await screen.findByDisplayValue("Engineering");

    await user.click(screen.getByRole("tab", { name: /MCP Servers/ }));
    await user.click(screen.getByRole("tab", { name: /Agents/ }));
    await user.click(screen.getByRole("tab", { name: /General Info/ }));
    await save(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(variables().params).toStrictEqual({
      access_group_name: "Engineering",
      description: "Engineers",
      access_model_names: undefined,
      access_mcp_server_ids: ["srv-1"],
      access_agent_ids: ["agent-1"],
    });
  });

  it("coerces a null description to an empty string", async () => {
    const user = setup();
    renderModal({ ...accessGroup, description: null });
    await screen.findByDisplayValue("Engineering");

    await save(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(variables().params.description).toBe("");
  });

  it("does not trim surrounding whitespace from the group name", async () => {
    const user = setup();
    renderModal();
    const nameInput = await screen.findByDisplayValue("Engineering");

    await user.clear(nameInput);
    fireEvent.change(nameInput, { target: { value: "  Padded  " } });
    await save(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(variables().params.access_group_name).toBe("  Padded  ");
  });

  it("never sends server-only fields from the loaded record", async () => {
    const user = setup();
    renderModal();
    await screen.findByDisplayValue("Engineering");

    await save(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(variables().params).not.toHaveProperty("access_group_id");
    expect(variables().params).not.toHaveProperty("created_at");
    expect(variables().params).not.toHaveProperty("assigned_team_ids");
    expect(variables().params).not.toHaveProperty("assigned_key_ids");
  });

  it("does not submit when the group name is cleared", async () => {
    const user = setup();
    renderModal();
    const nameInput = await screen.findByDisplayValue("Engineering");

    await user.clear(nameInput);
    await save(user);

    expect(await screen.findByText("Please enter the access group name")).toBeInTheDocument();
    expect(mutate).not.toHaveBeenCalled();
  });

  it("does not save when Enter is pressed in the name field", async () => {
    const user = setup();
    renderModal();
    const nameInput = await screen.findByDisplayValue("Engineering");

    await user.type(nameInput, "{Enter}");

    expect(mutate).not.toHaveBeenCalled();
  });

  it("sends models chosen on the Models tab", async () => {
    const user = setup();
    renderModal({ ...accessGroup, access_model_names: [] });
    await screen.findByDisplayValue("Engineering");

    await user.click(screen.getByRole("tab", { name: /Models/ }));
    await user.click(await screen.findByLabelText("model-select"));
    await save(user);

    await waitFor(() => expect(mutate).toHaveBeenCalled());
    expect(variables().params.access_model_names).toStrictEqual(["gpt-4"]);
  });
});
