import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import CreateKey from "./create_key_button";

const { mockKeyCreateCall } = vi.hoisted(() => ({
  mockKeyCreateCall: vi.fn().mockResolvedValue({ key: "sk-created", soft_budget: null }),
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "test-token", userId: "test-user-id", userRole: "Admin", premiumUser: true }),
}));

vi.mock("@/app/(dashboard)/hooks/useCan", () => ({ default: () => false }));

vi.mock("@/app/(dashboard)/hooks/keys/useKeys", () => ({ keyKeys: { lists: () => ["keys"] } }));

vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  useOrganizations: () => ({ data: [], isLoading: false }),
}));

vi.mock("@/app/(dashboard)/hooks/projects/useProjects", () => ({
  useProjects: () => ({ data: [], isLoading: false }),
}));

vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: () => ({ data: { values: {} } }),
}));

vi.mock("@/app/(dashboard)/hooks/tags/useTags", () => ({ useTags: () => ({ data: {} }) }));

vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useInfiniteTeams: () => ({ data: { pages: [{ teams: [] }] }, fetchNextPage: vi.fn(), hasNextPage: false }),
}));

vi.mock("@/app/(dashboard)/hooks/accessGroups/useAccessGroups", () => ({
  useAccessGroups: () => ({ data: [], isLoading: false, isError: false }),
}));

vi.mock("../networking", () => ({
  keyCreateCall: mockKeyCreateCall,
  keyCreateServiceAccountCall: vi.fn().mockResolvedValue({ key: "sk-sa", soft_budget: null }),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [{ id: "gpt-4" }] }),
  getGuardrailsList: vi.fn().mockResolvedValue({ guardrails: [] }),
  getPoliciesList: vi.fn().mockResolvedValue({ policies: [] }),
  getPromptsList: vi.fn().mockResolvedValue({ prompts: [] }),
  getPossibleUserRoles: vi.fn().mockResolvedValue({}),
  userFilterUICall: vi.fn().mockResolvedValue([]),
  fetchMCPAccessGroups: vi.fn().mockResolvedValue([]),
  getAgentsList: vi.fn().mockResolvedValue({ agents: [] }),
  getPassThroughEndpointsCall: vi.fn().mockResolvedValue({ endpoints: [] }),
  proxyBaseUrl: "http://localhost:4000",
}));

vi.mock("../agent_management/AgentSelector", () => ({ default: () => null }));
vi.mock("../common_components/check_openapi_schema", () => ({ default: () => null }));
vi.mock("../common_components/PremiumLoggingSettings", () => ({ default: () => null }));
vi.mock("../common_components/RouterSettingsAccordion", () => ({ default: () => null }));
vi.mock("../mcp_server_management/MCPServerSelector", () => ({ default: () => null }));
vi.mock("../mcp_server_management/MCPToolPermissions", () => ({ default: () => null }));
vi.mock("../vector_store_management/VectorStoreSelector", () => ({ default: () => null }));
vi.mock("../CreateUserButton", () => ({ CreateUserButton: () => null }));

describe("CreateKey submit payload contract", () => {
  beforeEach(() => {
    mockKeyCreateCall.mockClear();
  });

  const openModal = async () => {
    renderWithProviders(<CreateKey team={null} teams={[]} data={[]} addKey={() => {}} />);
    await userEvent.click(screen.getAllByTestId("create-key-button")[0]);
    await screen.findByRole("button", { name: /create key/i });
  };

  it("sends exactly the bound form fields for a minimal create", async () => {
    await openModal();

    await userEvent.type(screen.getByLabelText(/Key Name/), "probe-key");
    await userEvent.click(screen.getByRole("button", { name: /create key/i }));

    await waitFor(() => {
      expect(mockKeyCreateCall).toHaveBeenCalled();
    });
    expect(mockKeyCreateCall.mock.calls[0][2]).toStrictEqual({
      organization_id: undefined,
      team_id: null,
      key_alias: "probe-key",
      models: [],
      key_type: "llm_api",
      user_id: "test-user-id",
      duration: null,
      metadata: "{}",
    });
  });

  it("keeps a collapsed Optional Settings section out of the payload entirely", async () => {
    await openModal();

    await userEvent.type(screen.getByLabelText(/Key Name/), "probe-key");
    await userEvent.click(screen.getByRole("button", { name: /create key/i }));

    await waitFor(() => {
      expect(mockKeyCreateCall).toHaveBeenCalled();
    });
    const payload = mockKeyCreateCall.mock.calls[0][2];
    expect(payload).not.toHaveProperty("tpm_limit_type");
    expect(payload).not.toHaveProperty("rpm_limit_type");
    expect(payload).not.toHaveProperty("max_budget");
  });

  it("carries the shared rate-limit-type control into the payload once its section is open", async () => {
    await openModal();

    await userEvent.type(screen.getByLabelText(/Key Name/), "probe-key");
    await userEvent.click(screen.getByText("Optional Settings"));

    await userEvent.click(await screen.findByLabelText(/TPM Rate Limit Type/));
    await userEvent.click(await screen.findByText("Guaranteed throughput"));

    await userEvent.click(screen.getByRole("button", { name: /create key/i }));

    await waitFor(() => {
      expect(mockKeyCreateCall).toHaveBeenCalled();
    });
    expect(mockKeyCreateCall.mock.calls[0][2]).toMatchObject({
      tpm_limit_type: "guaranteed_throughput",
      rpm_limit_type: null,
    });
  });

  it("preserves a value typed in a section that is collapsed and reopened before submit", async () => {
    await openModal();

    await userEvent.type(screen.getByLabelText(/Key Name/), "probe-key");
    await userEvent.click(screen.getByText("Optional Settings"));

    const maxBudget = await screen.findByLabelText(/Max Budget/);
    await userEvent.type(maxBudget, "150.75");

    await userEvent.click(screen.getByText("Optional Settings"));
    await userEvent.click(screen.getByText("Optional Settings"));

    expect(await screen.findByLabelText(/Max Budget/)).toHaveValue(150.75);

    await userEvent.click(screen.getByRole("button", { name: /create key/i }));

    await waitFor(() => {
      expect(mockKeyCreateCall).toHaveBeenCalled();
    });
    expect(mockKeyCreateCall.mock.calls[0][2]).toMatchObject({ max_budget: "150.75" });
  });
});
