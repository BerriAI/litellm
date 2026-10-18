import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { toast } from "@/lib/toast";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import type { KeyResponse } from "../key_team_helpers/key_list";
import type { MCPServer } from "../mcp_tools/types";
import { NO_MCP_SERVERS_SENTINEL } from "../mcp_tools/constants";
import { toKeyEditFormValues, toSubmittedValues } from "./keyEditFormValues";

vi.mock("@/lib/toast", () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}));

// ---- Hoisted shared mocks (safe to use inside vi.mock factories) ----
const { keyUpdateCallMock, keyDeleteCallMock, invalidateQueriesMock, mockUseAuthorized } = vi.hoisted(() => {
  return {
    keyUpdateCallMock: vi.fn().mockResolvedValue({}),
    keyDeleteCallMock: vi.fn().mockResolvedValue({}),
    invalidateQueriesMock: vi.fn().mockResolvedValue(undefined),
    mockUseAuthorized: vi.fn(),
  };
});

// ---- Module mocks ----

// Mock useAuthorized hook FIRST (before component imports it)
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: mockUseAuthorized,
}));

vi.mock("@/app/(dashboard)/hooks/models/useModels", () => ({
  useModelAccessGroupNames: vi.fn(() => new Set<string>()),
}));

vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  useOrganizations: () => ({ data: [] }),
}));

vi.mock("@/app/(dashboard)/hooks/teams/useTeamMemberBudgets", () => ({
  useTeamMemberBudgets: vi.fn().mockReturnValue({}),
}));

// Networking: wire the hoisted fns so we can assert calls later
vi.mock("../networking", () => {
  return {
    serverRootPath: "",
    keyUpdateCall: (...args: any[]) => keyUpdateCallMock(...args),
    keyDeleteCall: (...args: any[]) => keyDeleteCallMock(...args),
  };
});

// Roles: ensure 'Admin' has write access and include all role helper functions
vi.mock("../../utils/roles", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../utils/roles")>();
  return {
    ...actual,
    rolesWithWriteAccess: ["Admin"],
  };
});

// Helpers used in rendering
vi.mock("@/utils/dataUtils", () => ({
  copyToClipboard: async () => true,
  formatNumberWithCommas: (n: any) => String(n),
}));
vi.mock("../key_info_utils", () => ({
  extractLoggingSettings: () => ({}),
  formatMetadataForDisplay: (m: any) => JSON.stringify(m, null, 2),
  stripTagsFromMetadata: (m: any) => m,
}));
vi.mock("../callback_info_helpers", () => ({
  callback_map: {},
  mapInternalToDisplayNames: (x: any) => x,
  mapDisplayToInternalNames: (x: any) => x,
}));
vi.mock("../shared/errorUtils", () => ({
  parseErrorMessage: (e: any) => String(e),
}));

// Icons -> async factory & local React
vi.mock("@heroicons/react/outline", async () => {
  const React = await import("react");
  function ArrowLeftIcon() {
    return React.createElement("span");
  }
  (ArrowLeftIcon as any).displayName = "ArrowLeftIcon";
  function TrashIcon() {
    return React.createElement("span");
  }
  (TrashIcon as any).displayName = "TrashIcon";
  function RefreshIcon() {
    return React.createElement("span");
  }
  (RefreshIcon as any).displayName = "RefreshIcon";
  return { ArrowLeftIcon, TrashIcon, RefreshIcon };
});

// Heavy children -> async factories & local React
vi.mock("../organisms/RegenerateKeyModal", () => {
  function RegenerateKeyModal() {
    return null;
  }
  (RegenerateKeyModal as any).displayName = "RegenerateKeyModal";
  return { RegenerateKeyModal };
});
vi.mock("../object_permissions_view", () => {
  function ObjectPermissionsView() {
    return null;
  }
  (ObjectPermissionsView as any).displayName = "ObjectPermissionsView";
  return { __esModule: true, default: ObjectPermissionsView };
});
vi.mock("../logging_settings_view", () => {
  function LoggingSettingsView() {
    return null;
  }
  (LoggingSettingsView as any).displayName = "LoggingSettingsView";
  return { __esModule: true, default: LoggingSettingsView };
});
vi.mock("../common_components/AutoRotationView", () => {
  function AutoRotationView() {
    return null;
  }
  (AutoRotationView as any).displayName = "AutoRotationView";
  return { __esModule: true, default: AutoRotationView };
});

// Mock Next.js router to avoid "invariant expected app router to be mounted" error
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    refresh: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
  }),
}));

// Mock useTeams hook
vi.mock("@/app/(dashboard)/hooks/useTeams", () => ({
  default: vi.fn(() => ({
    teams: [],
    setTeams: vi.fn(),
  })),
}));

// Mock useProjects hook
vi.mock("@/app/(dashboard)/hooks/projects/useProjects", () => ({
  useProjects: vi.fn().mockReturnValue({ data: [], isLoading: false }),
}));

// Mock useUISettings hook
vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: vi.fn().mockReturnValue({ data: { values: {} }, isLoading: false }),
}));

// Mock useMCPServers hook (requires QueryClientProvider which is not available in this test)
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPServers", () => ({
  useMCPServers: vi.fn().mockReturnValue({ data: [] }),
}));

vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPToolsets", () => ({
  useMCPToolsets: vi.fn().mockReturnValue({ data: [] }),
}));

// Mock useResetKeySpend hook (requires QueryClientProvider which is not available in this test)
vi.mock("@/app/(dashboard)/hooks/keys/useResetKeySpend", () => ({
  useResetKeySpend: vi.fn().mockReturnValue({
    mutate: vi.fn(),
    isPending: false,
  }),
}));

vi.mock("@/app/(dashboard)/hooks/keys/useSetKeyBlockedState", () => ({
  useSetKeyBlockedState: vi.fn().mockReturnValue({
    mutate: vi.fn(),
    isPending: false,
  }),
}));

// useQueryClient also needs a provider; the delete-path invalidation is covered in key_info_view.test.tsx
vi.mock("@tanstack/react-query", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@tanstack/react-query")>();
  return {
    ...actual,
    useQueryClient: () => ({ invalidateQueries: invalidateQueriesMock }),
  };
});

// KeyEditView mock: triggers onSubmit with our injected form values
vi.mock("./key_edit_view", async () => {
  const React = await import("react");
  function KeyEditView(props: any) {
    return React.createElement(
      "div",
      null,
      React.createElement(
        "button",
        {
          onClick: () => props.onSubmit((globalThis as any).__TEST_FORM_VALUES ?? {}),
        },
        "Mock Submit",
      ),
    );
  }
  (KeyEditView as any).displayName = "KeyEditViewMock";
  return { KeyEditView };
});

// ---- SUT import AFTER mocks ----
import KeyInfoView from "./key_info_view";

// ---- Test data helpers ----
const baseKeyData = {
  token_id: "tok_123",
  token: "tok_123",
  key_alias: "My Virtual Key",
  key_name: "sk-xxxx",
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  spend: 0,
  max_budget: null,
  tpm_limit: null,
  rpm_limit: null,
  models: [] as string[],
  metadata: {} as Record<string, any>,
  object_permission: {} as Record<string, any>,
  auto_rotate: false,
  rotation_interval: null as any,
  last_rotation_at: null as any,
  key_rotation_at: null as any,
  next_rotation_at: null as any,
};

const renderView = (premiumUser: boolean, keyData: KeyResponse = baseKeyData as KeyResponse) => {
  // Configure the mock for this test
  mockUseAuthorized.mockReturnValue({
    accessToken: "access_abc",
    userId: "user_1",
    userRole: "Admin",
    premiumUser,
    token: "token_123",
    userEmail: "test@example.com",
    disabledPersonalKeyCreation: false,
    showSSOBanner: false,
  });

  return render(
    <KeyInfoView keyId="tok_123" onClose={() => {}} keyData={keyData} onKeyDataUpdate={() => {}} teams={[]} />,
  );
};

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(useMCPServers).mockReturnValue({ data: [] } as ReturnType<typeof useMCPServers>);
  (globalThis as any).__TEST_FORM_VALUES = undefined;
});

// ---- Tests ----
describe("KeyInfoView handleKeyUpdate guardrails guard", () => {
  it("should remove guardrails & prompts for non-premium key owner without write access role", async () => {
    const keyDataWithOwner = { ...baseKeyData, user_id: "user_1" };
    mockUseAuthorized.mockReturnValue({
      accessToken: "access_abc",
      userId: "user_1",
      userRole: "viewer",
      premiumUser: false,
      token: "token_123",
      userEmail: "test@example.com",
      disabledPersonalKeyCreation: false,
      showSSOBanner: false,
    });

    render(
      <KeyInfoView
        keyId="tok_123"
        onClose={() => {}}
        keyData={keyDataWithOwner as any}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );

    fireEvent.click(screen.getByText("Settings"));
    fireEvent.click(screen.getByText("Edit Settings"));
    (globalThis as any).__TEST_FORM_VALUES = {
      token: "tok_123",
      guardrails: ["gr-1", "gr-2"],
      prompts: ["fast", "safe"],
      metadata: {},
    };

    fireEvent.click(screen.getByText("Mock Submit"));

    await waitFor(() => expect(keyUpdateCallMock).toHaveBeenCalled());

    const [sentAccessToken, sentPayload] = keyUpdateCallMock.mock.calls[0];
    expect(sentAccessToken).toBe("access_abc");

    expect("guardrails" in sentPayload).toBe(false);
    expect("prompts" in sentPayload).toBe(false);
    expect(sentPayload.metadata?.guardrails).toBeUndefined();
    expect(sentPayload.key).toBe("tok_123");
  });

  it("should preserve guardrails & prompts for non-premium users with write access role (e.g. Admin)", async () => {
    renderView(false); // premiumUser = false, userRole = "Admin"

    fireEvent.click(screen.getByText("Settings"));
    fireEvent.click(screen.getByText("Edit Settings"));
    (globalThis as any).__TEST_FORM_VALUES = {
      token: "tok_123",
      guardrails: ["gr-1"],
      prompts: ["fast"],
      metadata: {},
    };

    fireEvent.click(screen.getByText("Mock Submit"));

    await waitFor(() => expect(keyUpdateCallMock).toHaveBeenCalled());

    const [, sentPayload] = keyUpdateCallMock.mock.calls[0];

    expect(sentPayload.guardrails).toEqual(["gr-1"]);
    expect(sentPayload.prompts).toEqual(["fast"]);
    expect(sentPayload.metadata?.guardrails).toEqual(["gr-1"]);
    expect(sentPayload.key).toBe("tok_123");
  });

  it("should preserve guardrails & prompts for premium users and includes metadata.guardrails", async () => {
    renderView(true); // premiumUser = true

    fireEvent.click(screen.getByText("Settings"));
    fireEvent.click(screen.getByText("Edit Settings"));
    (globalThis as any).__TEST_FORM_VALUES = {
      token: "tok_123",
      guardrails: ["gr-1"],
      prompts: ["fast"],
      metadata: {},
    };

    fireEvent.click(screen.getByText("Mock Submit"));

    await waitFor(() => expect(keyUpdateCallMock).toHaveBeenCalled());

    const [, sentPayload] = keyUpdateCallMock.mock.calls[0];

    expect(sentPayload.guardrails).toEqual(["gr-1"]);
    expect(sentPayload.prompts).toEqual(["fast"]);
    expect(sentPayload.metadata?.guardrails).toEqual(["gr-1"]);
    expect(sentPayload.key).toBe("tok_123");
  });
});

describe("KeyInfoView handleKeyUpdate mcp_toolsets", () => {
  it("should forward the toolsets the edit form supplies into object_permission", async () => {
    renderView(true);

    fireEvent.click(screen.getByText("Settings"));
    fireEvent.click(screen.getByText("Edit Settings"));
    (globalThis as any).__TEST_FORM_VALUES = {
      token: "tok_123",
      max_budget: 40000,
      mcp_servers_and_groups: { servers: [], accessGroups: [], toolsets: ["ts-1"] },
    };

    fireEvent.click(screen.getByText("Mock Submit"));

    await waitFor(() => expect(keyUpdateCallMock).toHaveBeenCalled());

    const [, sentPayload] = keyUpdateCallMock.mock.calls[0];
    expect(sentPayload.object_permission.mcp_toolsets).toEqual(["ts-1"]);
    expect(sentPayload.max_budget).toBe(40000);
  });
});

describe("KeyInfoView handleKeyUpdate shared MCP aliases", () => {
  it.each([
    {
      edit: "only renaming the key",
      storedServers: ["server-a"],
      storedToolPermissions: { shared: ["read"] },
      selectedServers: ["server-a"],
      expectedToolPermissions: { shared: ["read"] },
    },
    {
      edit: "removing an explicitly selected server",
      storedServers: ["server-a", "server-b"],
      storedToolPermissions: { shared: ["read"] },
      selectedServers: ["server-a"],
      expectedToolPermissions: { "server-a": ["read"] },
    },
    {
      edit: "choosing No MCP Servers",
      storedServers: ["server-a"],
      storedToolPermissions: { "server-b": ["read"] },
      selectedServers: [NO_MCP_SERVERS_SENTINEL],
      expectedToolPermissions: {},
    },
  ])(
    "preserves unrelated permissions when $edit",
    async ({ storedServers, storedToolPermissions, selectedServers, expectedToolPermissions }) => {
      const servers: MCPServer[] = ["server-a", "server-b"].map((server_id) => ({
        server_id,
        alias: "shared",
        created_at: "",
        created_by: "",
        updated_at: "",
        updated_by: "",
      }));
      vi.mocked(useMCPServers).mockReturnValue({ data: servers } as ReturnType<typeof useMCPServers>);
      const permission = {
        mcp_servers: storedServers,
        mcp_access_groups: [],
        mcp_toolsets: [],
        mcp_tool_permissions: storedToolPermissions,
        vector_stores: ["vector-store"],
        agents: ["agent"],
        agent_access_groups: ["agent-group"],
        skills: ["skill"],
        search_tools: ["search-tool"],
      };
      const keyData = { ...baseKeyData, object_permission: permission } as KeyResponse;
      renderView(true, keyData);
      fireEvent.click(screen.getByText("Settings"));
      fireEvent.click(screen.getByText("Edit Settings"));
      (globalThis as { __TEST_FORM_VALUES?: Record<string, unknown> }).__TEST_FORM_VALUES = toSubmittedValues(
        {
          ...toKeyEditFormValues(keyData),
          key_alias: "Renamed key",
          mcp_servers_and_groups: { servers: selectedServers, accessGroups: [], toolsets: [] },
        },
        { canViewPolicies: true, canViewPrompts: true },
      );

      fireEvent.click(screen.getByText("Mock Submit"));

      await waitFor(() =>
        expect(keyUpdateCallMock).toHaveBeenCalledExactlyOnceWith(
          "access_abc",
          expect.objectContaining({
            key: keyData.token,
            key_alias: "Renamed key",
            object_permission: {
              ...permission,
              mcp_servers: selectedServers,
              mcp_tool_permissions: expectedToolPermissions,
            },
          }),
        ),
      );
    },
  );
});

describe("KeyInfoView handleKeyUpdate cache sync", () => {
  it("should invalidate every cached key query so the list and detail views re-read the saved key", async () => {
    keyUpdateCallMock.mockResolvedValueOnce({
      object_permission: { mcp_servers: ["srv-1"], mcp_tool_permissions: { "srv-1": ["read_wiki"] } },
    });
    renderView(true);

    fireEvent.click(screen.getByText("Settings"));
    fireEvent.click(screen.getByText("Edit Settings"));
    (globalThis as any).__TEST_FORM_VALUES = { token: "tok_123", metadata: {} };

    fireEvent.click(screen.getByText("Mock Submit"));

    await waitFor(() => expect(toast.success).toHaveBeenCalledWith("Key updated successfully"));
    expect(invalidateQueriesMock).toHaveBeenCalledWith({ queryKey: ["keys"] });
  });
});

describe("KeyInfoView handleKeyUpdate skills", () => {
  it("should forward the skills the edit form supplies into object_permission and drop the form key", async () => {
    renderView(true);

    fireEvent.click(screen.getByText("Settings"));
    fireEvent.click(screen.getByText("Edit Settings"));
    (globalThis as any).__TEST_FORM_VALUES = {
      token: "tok_123",
      skills: ["private-skill"],
    };

    fireEvent.click(screen.getByText("Mock Submit"));

    await waitFor(() => expect(keyUpdateCallMock).toHaveBeenCalled());

    const [, sentPayload] = keyUpdateCallMock.mock.calls[0];
    expect(sentPayload.object_permission.skills).toEqual(["private-skill"]);
    expect(sentPayload).not.toHaveProperty("skills");
  });

  it("should send an explicit empty skills list when the form clears every skill", async () => {
    renderView(true);

    fireEvent.click(screen.getByText("Settings"));
    fireEvent.click(screen.getByText("Edit Settings"));
    (globalThis as any).__TEST_FORM_VALUES = {
      token: "tok_123",
      skills: [],
    };

    fireEvent.click(screen.getByText("Mock Submit"));

    await waitFor(() => expect(keyUpdateCallMock).toHaveBeenCalled());

    const [, sentPayload] = keyUpdateCallMock.mock.calls[0];
    expect(sentPayload.object_permission.skills).toEqual([]);
  });
});

describe("KeyInfoView handleKeyUpdate budget_duration", () => {
  it("should send a canonical budget_duration through unchanged", async () => {
    renderView(true);

    fireEvent.click(screen.getByText("Settings"));
    fireEvent.click(screen.getByText("Edit Settings"));
    (globalThis as any).__TEST_FORM_VALUES = {
      token: "tok_123",
      budget_duration: "30d",
    };

    fireEvent.click(screen.getByText("Mock Submit"));

    await waitFor(() => expect(keyUpdateCallMock).toHaveBeenCalled());

    const [, sentPayload] = keyUpdateCallMock.mock.calls[0];
    expect(sentPayload.budget_duration).toBe("30d");
  });

  it("should heal a legacy word-form budget_duration to canonical", async () => {
    renderView(true);

    fireEvent.click(screen.getByText("Settings"));
    fireEvent.click(screen.getByText("Edit Settings"));
    (globalThis as any).__TEST_FORM_VALUES = {
      token: "tok_123",
      budget_duration: "monthly",
    };

    fireEvent.click(screen.getByText("Mock Submit"));

    await waitFor(() => expect(keyUpdateCallMock).toHaveBeenCalled());

    const [, sentPayload] = keyUpdateCallMock.mock.calls[0];
    expect(sentPayload.budget_duration).toBe("30d");
  });

  it("should forward a cleared budget_duration as an explicit null the JSON body keeps", async () => {
    renderView(true);

    fireEvent.click(screen.getByText("Settings"));
    fireEvent.click(screen.getByText("Edit Settings"));
    (globalThis as any).__TEST_FORM_VALUES = {
      token: "tok_123",
      budget_duration: null,
    };

    fireEvent.click(screen.getByText("Mock Submit"));

    await waitFor(() => expect(keyUpdateCallMock).toHaveBeenCalled());

    const [, sentPayload] = keyUpdateCallMock.mock.calls[0];
    expect(sentPayload.budget_duration).toBeNull();
    expect(JSON.stringify({ ...sentPayload })).toContain('"budget_duration":null');
  });

  it("should render the cleared budget as never resetting instead of snapping back to the old interval", async () => {
    keyUpdateCallMock.mockResolvedValueOnce({
      ...baseKeyData,
      budget_duration: null,
      budget_reset_at: null,
    });
    mockUseAuthorized.mockReturnValue({
      accessToken: "access_abc",
      userId: "user_1",
      userRole: "Admin",
      premiumUser: true,
      token: "token_123",
      userEmail: "test@example.com",
      disabledPersonalKeyCreation: false,
      showSSOBanner: false,
    });

    render(
      <KeyInfoView
        keyId="tok_123"
        onClose={() => {}}
        keyData={{ ...baseKeyData, budget_duration: "30d", budget_reset_at: "2026-09-01T00:00:00Z" } as any}
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );

    fireEvent.click(screen.getByText("Settings"));
    expect(screen.getByTestId("budget-reset-value")).toHaveTextContent("Every 30d");

    fireEvent.click(screen.getByText("Edit Settings"));
    (globalThis as any).__TEST_FORM_VALUES = {
      token: "tok_123",
      budget_duration: null,
    };

    fireEvent.click(screen.getByText("Mock Submit"));

    await waitFor(() => {
      expect(screen.getByTestId("budget-reset-value")).toHaveTextContent("Never");
    });
  });
});

describe("KeyInfoView handleKeyUpdate empty strings", () => {
  ["tpm_limit", "rpm_limit", "max_parallel_requests", "max_budget"].forEach((limit) => {
    it(`maps empty strings to null for ${limit}`, async () => {
      renderView(true); // premiumUser = true

      fireEvent.click(screen.getByText("Settings"));
      fireEvent.click(screen.getByText("Edit Settings"));
      (globalThis as any).__TEST_FORM_VALUES = {
        token: "tok_123",
        [limit]: "",
      };

      fireEvent.click(screen.getByText("Mock Submit"));

      await waitFor(() => expect(keyUpdateCallMock).toHaveBeenCalled());

      const [sentAccessToken, sentPayload] = keyUpdateCallMock.mock.calls[0];
      expect(sentAccessToken).toBe("access_abc");
      expect(sentPayload[limit]).toBeNull();
    });
  });
});

describe("KeyInfoView handleKeyUpdate soft_budget", () => {
  const premiumAdminAuth = {
    accessToken: "access_abc",
    userId: "user_1",
    userRole: "Admin",
    premiumUser: true,
    token: "token_123",
    userEmail: "test@example.com",
    disabledPersonalKeyCreation: false,
    showSSOBanner: false,
  };

  const renderWithSoftBudget = (softBudget: number | null) => {
    mockUseAuthorized.mockReturnValue(premiumAdminAuth);

    return render(
      <KeyInfoView
        keyId="tok_123"
        onClose={() => {}}
        keyData={
          { ...baseKeyData, litellm_budget_table: softBudget === null ? null : { soft_budget: softBudget } } as any
        }
        onKeyDataUpdate={() => {}}
        teams={[]}
      />,
    );
  };

  it("should send a changed soft_budget as a number", async () => {
    renderWithSoftBudget(null);

    fireEvent.click(screen.getByText("Settings"));
    fireEvent.click(screen.getByText("Edit Settings"));
    (globalThis as any).__TEST_FORM_VALUES = {
      token: "tok_123",
      soft_budget: "25",
    };

    fireEvent.click(screen.getByText("Mock Submit"));

    await waitFor(() => expect(keyUpdateCallMock).toHaveBeenCalled());

    const [, sentPayload] = keyUpdateCallMock.mock.calls[0];
    expect(sentPayload.soft_budget).toBe(25);
  });

  it("should omit an unchanged soft_budget so unrelated edits skip the budget gate", async () => {
    renderWithSoftBudget(25);

    fireEvent.click(screen.getByText("Settings"));
    fireEvent.click(screen.getByText("Edit Settings"));
    (globalThis as any).__TEST_FORM_VALUES = {
      token: "tok_123",
      soft_budget: 25,
      key_alias: "renamed",
    };

    fireEvent.click(screen.getByText("Mock Submit"));

    await waitFor(() => expect(keyUpdateCallMock).toHaveBeenCalled());

    const [, sentPayload] = keyUpdateCallMock.mock.calls[0];
    expect("soft_budget" in sentPayload).toBe(false);
  });

  it("should forward a cleared soft_budget as an explicit null the JSON body keeps", async () => {
    renderWithSoftBudget(25);

    fireEvent.click(screen.getByText("Settings"));
    fireEvent.click(screen.getByText("Edit Settings"));
    (globalThis as any).__TEST_FORM_VALUES = {
      token: "tok_123",
      soft_budget: "",
    };

    fireEvent.click(screen.getByText("Mock Submit"));

    await waitFor(() => expect(keyUpdateCallMock).toHaveBeenCalled());

    const [, sentPayload] = keyUpdateCallMock.mock.calls[0];
    expect(sentPayload.soft_budget).toBeNull();
    expect(JSON.stringify({ ...sentPayload })).toContain('"soft_budget":null');
  });

  it("should reject an overflowing soft_budget instead of silently clearing it", async () => {
    renderWithSoftBudget(25);

    fireEvent.click(screen.getByText("Settings"));
    fireEvent.click(screen.getByText("Edit Settings"));
    (globalThis as any).__TEST_FORM_VALUES = {
      token: "tok_123",
      soft_budget: "1e309",
    };

    fireEvent.click(screen.getByText("Mock Submit"));

    await waitFor(() => expect(toast.error).toHaveBeenCalled());
    expect(keyUpdateCallMock).not.toHaveBeenCalled();
  });
});
