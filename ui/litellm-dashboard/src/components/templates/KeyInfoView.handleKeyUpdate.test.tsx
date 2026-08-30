import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

// ---- Hoisted shared mocks (safe to use inside vi.mock factories) ----
const { keyUpdateCallMock, keyDeleteCallMock, mockUseAuthorized } = vi.hoisted(() => {
  return {
    keyUpdateCallMock: vi.fn().mockResolvedValue({}),
    keyDeleteCallMock: vi.fn().mockResolvedValue({}),
    mockUseAuthorized: vi.fn(),
  };
});

// ---- Module mocks ----

// Mock useAuthorized hook FIRST (before component imports it)
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: mockUseAuthorized,
}));

vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  useOrganizations: () => ({ data: [] }),
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
    useQueryClient: () => ({ invalidateQueries: vi.fn() }),
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

const renderView = (premiumUser: boolean) => {
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
    <KeyInfoView
      keyId="tok_123"
      onClose={() => {}}
      keyData={baseKeyData as any}
      onKeyDataUpdate={() => {}}
      teams={[]}
    />,
  );
};

beforeEach(() => {
  vi.clearAllMocks();
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
    expect(screen.getByText("Budget Reset").parentElement?.textContent).toContain("Every 30d");

    fireEvent.click(screen.getByText("Edit Settings"));
    (globalThis as any).__TEST_FORM_VALUES = {
      token: "tok_123",
      budget_duration: null,
    };

    fireEvent.click(screen.getByText("Mock Submit"));

    await waitFor(() => {
      expect(screen.getByText("Budget Reset").parentElement?.textContent).toBe("Budget ResetNever");
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
