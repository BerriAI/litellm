import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, renderWithProviders, screen, testQueryClient, waitFor } from "../../../tests/test-utils";
import type { Team } from "../key_team_helpers/key_list";
import { keyCreateCall, keyCreateServiceAccountCall, modelAvailableCall, userFilterUICall } from "../networking";
import { toast } from "@/lib/toast";
import CreateKey from "./create_key_button";

const state = vi.hoisted(() => ({
  authorized: {
    accessToken: "test-token",
    userId: "test-user-id",
    userRole: "Admin",
    premiumUser: false,
  },
  can: {} as Record<string, boolean>,
  uiSettings: {} as Record<string, unknown>,
  tags: {} as Record<string, { name: string }>,
  teams: [] as { team_id: string; team_alias: string; models: string[] }[],
  organizations: [] as { organization_id: string; organization_alias: string }[],
  accessGroups: [] as { access_group_id: string; access_group_name: string }[],
  projects: [] as { project_id: string; project_alias: string; team_id?: string; models?: string[] }[],
}));

vi.mock("@/lib/toast", () => ({
  toast: {
    success: vi.fn(),
    info: vi.fn(),
    warning: vi.fn(),
    error: vi.fn(),
    fromError: vi.fn(),
    dismiss: vi.fn(),
  },
}));
vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => state.authorized }));
vi.mock("@/app/(dashboard)/hooks/useCan", () => ({
  default: (capability: string) => state.can[capability] ?? true,
}));
vi.mock("@/app/(dashboard)/hooks/organizations/useOrganizations", () => ({
  useOrganizations: () => ({ data: state.organizations, isLoading: false }),
}));
vi.mock("@/app/(dashboard)/hooks/projects/useProjects", () => ({
  useProjects: () => ({ data: state.projects, isLoading: false }),
}));
vi.mock("@/app/(dashboard)/hooks/uiSettings/useUISettings", () => ({
  useUISettings: () => ({ data: { values: state.uiSettings } }),
}));
vi.mock("@/app/(dashboard)/hooks/tags/useTags", () => ({
  useTags: () => ({ data: state.tags, isLoading: false }),
}));
vi.mock("@/app/(dashboard)/hooks/teams/useTeams", () => ({
  useInfiniteTeams: () => ({
    data: { pages: [{ teams: state.teams, total: state.teams.length, page: 1, page_size: 50, total_pages: 1 }] },
    fetchNextPage: vi.fn(),
    hasNextPage: false,
    isFetchingNextPage: false,
    isLoading: false,
  }),
}));
vi.mock("@/app/(dashboard)/hooks/accessGroups/useAccessGroups", () => ({
  useAccessGroups: () => ({ data: state.accessGroups, isLoading: false, isError: false }),
}));
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPServers", () => ({
  useMCPServers: () => ({ data: [], isLoading: false }),
}));
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPAccessGroups", () => ({
  useMCPAccessGroups: () => ({ data: [], isLoading: false }),
}));
vi.mock("@/app/(dashboard)/hooks/mcpServers/useMCPToolsets", () => ({
  useMCPToolsets: () => ({ data: [], isLoading: false }),
}));

vi.mock("../networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../networking")>();
  const emptyMcpTools = { tools: [], error: null, message: null, stack_trace: null };
  return {
    ...actual,
    keyCreateCall: vi.fn().mockResolvedValue({ key: "sk-created", soft_budget: null }),
    keyCreateServiceAccountCall: vi.fn().mockResolvedValue({ key: "sk-service-account", soft_budget: null }),
    modelAvailableCall: vi.fn().mockResolvedValue({ data: [{ id: "gpt-4" }] }),
    getGuardrailsList: vi.fn().mockResolvedValue({ guardrails: [] }),
    getPoliciesList: vi.fn().mockResolvedValue({ policies: [] }),
    getPromptsList: vi.fn().mockResolvedValue({ prompts: [] }),
    getPossibleUserRoles: vi.fn().mockResolvedValue({}),
    userFilterUICall: vi.fn().mockResolvedValue([]),
    getAgentsList: vi.fn().mockResolvedValue({ agents: [] }),
    getPassThroughEndpointsCall: vi.fn().mockResolvedValue({ endpoints: [] }),
    vectorStoreListCall: vi.fn().mockResolvedValue({ data: [] }),
    listMCPTools: vi.fn().mockResolvedValue(emptyMcpTools),
    getRouterSettingsCall: vi.fn().mockResolvedValue({ router_settings: {} }),
  };
});

const OPENAPI_SCHEMA = {
  components: {
    schemas: {
      GenerateKeyRequest: {
        properties: {
          key: { type: "string", title: "Key" },
          soft_budget: { type: "number", title: "Soft Budget" },
          blocked: { type: "boolean", title: "Blocked" },
          max_budget: { type: "number", title: "Max Budget" },
        },
      },
    },
  },
};

const SECTIONS = {
  mcp: /MCP Settings/i,
  agent: /Agent Settings/i,
  logging: /Logging Settings/i,
  router: /Router Settings/i,
  aliases: /Model Aliases/i,
  lifecycle: /Key Lifecycle/i,
  advanced: /Advanced Settings/i,
} as const;

const ALL_CLOSED_PAYLOAD = {
  organization_id: undefined,
  team_id: null,
  key_alias: "contract-key",
  models: [],
  key_type: "llm_api",
  user_id: "test-user-id",
  duration: null,
  metadata: "{}",
};

const OPTIONAL_OPEN_PAYLOAD = {
  ...ALL_CLOSED_PAYLOAD,
  max_budget: undefined,
  budget_duration: undefined,
  tpm_limit: undefined,
  tpm_limit_type: null,
  rpm_limit: undefined,
  rpm_limit_type: null,
  throttle_on_budget_exceeded: undefined,
  enable_prompt_caching: undefined,
  guardrails: undefined,
  policies: undefined,
  prompts: undefined,
  access_group_ids: undefined,
  allowed_passthrough_routes: undefined,
  allowed_vector_store_ids: undefined,
  tags: undefined,
};

const ROUTER_SETTINGS_DEFAULT = {
  routing_strategy: null,
  allowed_fails: null,
  cooldown_time: null,
  num_retries: null,
  timeout: null,
  retry_after: null,
  fallbacks: null,
  context_window_fallbacks: null,
  retry_policy: null,
  model_group_alias: null,
  enable_tag_filtering: false,
  routing_strategy_args: null,
};

const SECTION_PAYLOAD_ADDITIONS: Record<keyof typeof SECTIONS, Record<string, unknown>> = {
  mcp: { allowed_mcp_servers_and_groups: { servers: [], accessGroups: [] } },
  agent: { allowed_agents_and_groups: undefined },
  logging: {},
  router: { router_settings: ROUTER_SETTINGS_DEFAULT },
  aliases: {},
  lifecycle: {},
  advanced: { key: undefined, soft_budget: undefined, blocked: undefined },
};

const ALL_OPEN_PAYLOAD = {
  ...OPTIONAL_OPEN_PAYLOAD,
  ...SECTION_PAYLOAD_ADDITIONS.mcp,
  ...SECTION_PAYLOAD_ADDITIONS.agent,
  ...SECTION_PAYLOAD_ADDITIONS.router,
  ...SECTION_PAYLOAD_ADDITIONS.advanced,
};

const renderCreateKey = (props: Partial<React.ComponentProps<typeof CreateKey>> = {}) =>
  renderWithProviders(<CreateKey team={null} teams={[]} data={[]} addKey={vi.fn()} {...props} />);

const openModal = async (props: Partial<React.ComponentProps<typeof CreateKey>> = {}) => {
  const view = renderCreateKey(props);
  await userEvent.click(screen.getByTestId("create-key-button"));
  await screen.findByRole("button", { name: /^create key$/i });
  return view;
};

const userSearchInput = (): Promise<HTMLElement> => screen.findByPlaceholderText("Type email to search for users");

const openSection = async (name: RegExp) => {
  await userEvent.click(await screen.findByRole("button", { name }));
};

const nameTheKey = async (alias = "contract-key") => {
  await userEvent.type(await screen.findByLabelText(/Key Name/), alias);
};

const submit = async () => {
  await userEvent.click(screen.getByRole("button", { name: /^create key$/i }));
};

const createdPayload = async () => {
  await waitFor(() => {
    expect(vi.mocked(keyCreateCall)).toHaveBeenCalled();
  });
  return vi.mocked(keyCreateCall).mock.calls[0][2] as Record<string, unknown>;
};

describe("CreateKey", () => {
  beforeEach(() => {
    testQueryClient.clear();
    state.authorized = { accessToken: "test-token", userId: "test-user-id", userRole: "Admin", premiumUser: false };
    state.can = {};
    state.uiSettings = {};
    state.tags = {};
    state.teams = [];
    state.organizations = [];
    state.projects = [];
    state.accessGroups = [];
    vi.mocked(keyCreateCall).mockClear().mockResolvedValue({ key: "sk-created", soft_budget: null });
    vi.mocked(keyCreateServiceAccountCall)
      .mockClear()
      .mockResolvedValue({ key: "sk-service-account", soft_budget: null });
    vi.mocked(userFilterUICall).mockClear().mockResolvedValue([]);
    vi.mocked(toast.fromError).mockClear();
    vi.mocked(modelAvailableCall)
      .mockClear()
      .mockResolvedValue({ data: [{ id: "gpt-4" }] });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input);
        if (url.includes("openapi.json")) {
          return { ok: true, status: 200, json: async () => OPENAPI_SCHEMA } as unknown as Response;
        }
        return { ok: true, status: 200, json: async () => ({}) } as unknown as Response;
      }),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  describe("submit payload contract", () => {
    it("sends only the always-mounted fields when every collapsible section is closed", async () => {
      await openModal();
      await nameTheKey();
      await submit();

      expect(await createdPayload()).toStrictEqual(ALL_CLOSED_PAYLOAD);
    });

    it("registers the Optional Settings fields as undefined-valued keys once that section is open", async () => {
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await submit();

      expect(await createdPayload()).toStrictEqual(OPTIONAL_OPEN_PAYLOAD);
    });

    it("sends the full mounted field set when every section is open", async () => {
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      for (const trigger of Object.values(SECTIONS)) {
        await openSection(trigger);
      }
      await screen.findByLabelText("Soft Budget");
      await submit();

      expect(await createdPayload()).toStrictEqual(ALL_OPEN_PAYLOAD);
    });

    it.each(Object.keys(SECTIONS) as (keyof typeof SECTIONS)[])(
      "adds exactly the %s section's own keys when it is the only nested section open",
      async (section) => {
        await openModal();
        await nameTheKey();
        await openSection(/Optional Settings/i);
        await openSection(SECTIONS[section]);
        if (section === "advanced") {
          await screen.findByLabelText("Soft Budget");
        }
        await submit();

        expect(await createdPayload()).toStrictEqual({
          ...OPTIONAL_OPEN_PAYLOAD,
          ...SECTION_PAYLOAD_ADDITIONS[section],
        });
      },
    );

    it.each([
      [
        "every section closed",
        false,
        ["team_id", "key_alias", "models", "key_type", "user_id", "duration", "metadata"],
      ],
      [
        "Optional Settings open",
        true,
        [
          "team_id",
          "key_alias",
          "models",
          "key_type",
          "tpm_limit_type",
          "rpm_limit_type",
          "user_id",
          "duration",
          "metadata",
        ],
      ],
    ])("serialises to exactly the wire keys with %s", async (_label, openOptional, wireKeys) => {
      await openModal();
      await nameTheKey();
      if (openOptional) {
        await openSection(/Optional Settings/i);
      }
      await submit();

      const serialised = JSON.parse(JSON.stringify(await createdPayload())) as Record<string, unknown>;
      expect(Object.keys(serialised).sort()).toStrictEqual([...wireKeys].sort());
    });

    it("omits a budget typed into a section the user closed again, rather than sending it as null", async () => {
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await userEvent.type(await screen.findByLabelText(/Max Budget \(USD\)/), "150.75");
      await openSection(/Optional Settings/i);
      await submit();

      const payload = await createdPayload();
      expect(payload).not.toHaveProperty("max_budget");
      expect(payload).toStrictEqual(ALL_CLOSED_PAYLOAD);
    });

    it("restores the typed budget when the closed section is expanded again before submit", async () => {
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await userEvent.type(await screen.findByLabelText(/Max Budget \(USD\)/), "150.75");
      await openSection(/Optional Settings/i);
      await openSection(/Optional Settings/i);

      expect(await screen.findByLabelText(/Max Budget \(USD\)/)).toHaveValue(150.75);

      await submit();

      expect(await createdPayload()).toStrictEqual({ ...OPTIONAL_OPEN_PAYLOAD, max_budget: "150.75" });
    });

    it("sends a typed max budget as a string, not a number", async () => {
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await userEvent.type(await screen.findByLabelText(/Max Budget \(USD\)/), "150.75");
      await submit();

      const payload = await createdPayload();
      expect(payload.max_budget).toBe("150.75");
    });

    it.each([
      ["Tokens per minute Limit (TPM)", "tpm_limit"],
      ["Requests per minute Limit (RPM)", "rpm_limit"],
    ])("routes a typed %s into the %s payload key", async (label, key) => {
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await userEvent.type(await screen.findByLabelText(label), "42");
      await submit();

      expect((await createdPayload())[key]).toBe("42");
    });

    it("routes the shared rate-limit-type control into its own payload key", async () => {
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await userEvent.click(await screen.findByLabelText(/TPM Rate Limit Type/));
      await userEvent.click(await screen.findByRole("option", { name: /Guaranteed throughput/ }));
      await submit();

      const payload = await createdPayload();
      expect(payload.tpm_limit_type).toBe("guaranteed_throughput");
      expect(payload.rpm_limit_type).toBeNull();
    });

    it("routes a typed expiry into duration, which is otherwise coalesced to null", async () => {
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await openSection(SECTIONS.lifecycle);
      await userEvent.type(await screen.findByLabelText("Expire Key"), "45d");
      await submit();

      expect((await createdPayload()).duration).toBe("45d");
    });

    it("drops a typed expiry back to null when its section is collapsed before submit", async () => {
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await openSection(SECTIONS.lifecycle);
      await userEvent.type(await screen.findByLabelText("Expire Key"), "45d");
      await openSection(SECTIONS.lifecycle);
      await submit();

      expect((await createdPayload()).duration).toBeNull();
    });

    it("routes a selected budget reset window into budget_duration", async () => {
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await userEvent.click(await screen.findByLabelText("Reset Budget"));
      await userEvent.click(await screen.findByRole("option", { name: "daily" }));
      await submit();

      expect((await createdPayload()).budget_duration).toBe("24h");
    });

    it("sends an explicit null budget_duration when the never-resets window is chosen", async () => {
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await userEvent.click(await screen.findByLabelText("Reset Budget"));
      await userEvent.click(await screen.findByRole("option", { name: /never resets/i }));
      await submit();

      const payload = await createdPayload();
      expect("budget_duration" in payload).toBe(true);
      expect(payload.budget_duration).toBeNull();
    });

    it("routes typed tags into the tags key", async () => {
      state.tags = { production: { name: "production" } };
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await userEvent.type(await screen.findByLabelText("Tags"), "production{Enter}");
      await submit();

      expect((await createdPayload()).tags).toStrictEqual(["production"]);
    });

    it("routes a chosen access group into access_group_ids", async () => {
      state.accessGroups = [{ access_group_id: "ag-1", access_group_name: "Group One" }];
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);

      await userEvent.click(await screen.findByLabelText("Select access groups (optional)"));
      await userEvent.click(await screen.findByRole("option", { name: /Group One/ }));
      await userEvent.keyboard("{Escape}");
      await submit();

      expect((await createdPayload()).access_group_ids).toStrictEqual(["ag-1"]);
    });

    it("moves the schema-driven Advanced Settings fields onto the payload under their own keys", async () => {
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await openSection(SECTIONS.advanced);
      await userEvent.type(await screen.findByLabelText("Soft Budget"), "12");
      await submit();

      const payload = await createdPayload();
      expect(payload.soft_budget).toBe(12);
      expect(payload).toHaveProperty("key");
    });

    it("drops the schema-driven custom key field when the proxy disables custom API keys", async () => {
      state.uiSettings = { disable_custom_api_keys: true };
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await openSection(SECTIONS.advanced);
      await screen.findByLabelText("Soft Budget");
      await submit();

      const payload = await createdPayload();
      expect(payload).not.toHaveProperty("key");
      expect(payload).toHaveProperty("soft_budget");
    });

    it("drops the policy and prompt keys entirely for a role that cannot see those fields", async () => {
      state.can = { viewPolicies: false, viewPrompts: false };
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await submit();

      const payload = await createdPayload();
      expect(payload).not.toHaveProperty("policies");
      expect(payload).not.toHaveProperty("prompts");
      expect(payload).toHaveProperty("guardrails");
    });

    it("adds project_id only when the projects UI is enabled", async () => {
      state.uiSettings = { enable_projects_ui: true };
      await openModal();
      await nameTheKey();
      await submit();

      expect(await createdPayload()).toStrictEqual({ ...ALL_CLOSED_PAYLOAD, project_id: undefined });
    });

    it("keeps disable_global_guardrails out of the payload while the switch is off", async () => {
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await submit();

      expect(await createdPayload()).not.toHaveProperty("disable_global_guardrails");
    });

    it("sends disable_global_guardrails once the switch is on", async () => {
      state.authorized = { ...state.authorized, premiumUser: true };
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await userEvent.click(await screen.findByRole("switch", { name: /Disable Global Guardrails/ }));
      await submit();

      expect((await createdPayload()).disable_global_guardrails).toBe(true);
    });

    it("folds a metadata JSON string back through JSON.stringify", async () => {
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await userEvent.type(await screen.findByLabelText("Metadata"), '{{"team":"research"}');
      await submit();

      expect((await createdPayload()).metadata).toBe('{"team":"research"}');
    });
  });

  describe("key ownership", () => {
    it("stamps the signed-in user onto user_id when the key is owned by you", async () => {
      await openModal();
      await nameTheKey();
      await submit();

      expect((await createdPayload()).user_id).toBe("test-user-id");
    });

    it("mounts the user search control only once Another User is chosen", async () => {
      await openModal();
      expect(screen.queryByPlaceholderText("Type email to search for users")).not.toBeInTheDocument();

      await userEvent.click(screen.getByRole("radio", { name: "Another User" }));

      expect(await userSearchInput()).toBeInTheDocument();
    });

    it("hides the Another User option from a non-admin", async () => {
      state.authorized = { ...state.authorized, userRole: "Internal User" };
      await openModal();

      expect(screen.queryByRole("radio", { name: "Another User" })).not.toBeInTheDocument();
    });

    it("routes a service account through the service account endpoint and stamps the alias into metadata", async () => {
      state.teams = [{ team_id: "team-1", team_alias: "Team One", models: [] }];
      await openModal({ teams: state.teams as unknown as Team[] });
      await userEvent.click(screen.getByRole("radio", { name: "Service Account" }));
      await userEvent.type(await screen.findByLabelText(/Service Account ID/), "svc-account-1");

      await userEvent.click(await screen.findByLabelText("Team"));
      await userEvent.click(await screen.findByRole("option", { name: /Team One/ }));

      await submit();

      await waitFor(() => {
        expect(vi.mocked(keyCreateServiceAccountCall)).toHaveBeenCalled();
      });
      expect(vi.mocked(keyCreateCall)).not.toHaveBeenCalled();
      const payload = vi.mocked(keyCreateServiceAccountCall).mock.calls[0][1] as Record<string, unknown>;
      expect(JSON.parse(String(payload.metadata))).toStrictEqual({ service_account_id: "svc-account-1" });
      expect(payload).not.toHaveProperty("user_id");
    });
  });

  describe("required field validation", () => {
    it("blocks the submit and marks the alias invalid when it is blank", async () => {
      await openModal();
      await submit();

      await waitFor(() => {
        expect(screen.getByLabelText(/Key Name/)).toHaveAttribute("aria-invalid", "true");
      });
      expect(vi.mocked(keyCreateCall)).not.toHaveBeenCalled();
    });

    it("suppresses the required message behind the always-visible help text", async () => {
      await openModal();
      await submit();

      await waitFor(() => {
        expect(screen.getByLabelText(/Key Name/)).toHaveAttribute("aria-invalid", "true");
      });
      expect(screen.queryByText("Please input a key name")).not.toBeInTheDocument();
      expect(screen.getByText("required")).toBeInTheDocument();
    });

    it("blocks a service account submit until a team is chosen, then lets it through", async () => {
      state.teams = [{ team_id: "team-1", team_alias: "Team One", models: [] }];
      await openModal({ teams: state.teams as unknown as Team[] });
      await userEvent.click(screen.getByRole("radio", { name: "Service Account" }));
      await userEvent.type(await screen.findByLabelText(/Service Account ID/), "svc-account-1");
      await submit();

      await waitFor(() => {
        expect(vi.mocked(keyCreateServiceAccountCall)).not.toHaveBeenCalled();
      });

      await userEvent.click(await screen.findByLabelText("Team"));
      await userEvent.click(await screen.findByRole("option", { name: /Team One/ }));
      await submit();

      await waitFor(() => {
        expect(vi.mocked(keyCreateServiceAccountCall)).toHaveBeenCalledTimes(1);
      });
    });

    it("blocks the submit when the budget exceeds the team ceiling", async () => {
      await openModal({ team: { team_id: "team-1", max_budget: 10 } as unknown as Team });
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await userEvent.type(await screen.findByLabelText(/Max Budget \(USD\)/), "50");
      await submit();

      await waitFor(() => {
        expect(screen.getByLabelText(/Max Budget \(USD\)/)).toHaveAttribute("aria-invalid", "true");
      });
      expect(vi.mocked(keyCreateCall)).not.toHaveBeenCalled();
    });
  });

  describe("deep link prefill", () => {
    it("prefills the key alias", async () => {
      renderCreateKey({ autoOpenCreate: true, prefillData: { key_alias: "prefilled-key" } });

      expect(await screen.findByLabelText(/Key Name/)).toHaveValue("prefilled-key");
    });

    it("prefills models once the available model list arrives", async () => {
      renderCreateKey({ autoOpenCreate: true, prefillData: { models: ["gpt-4"] } });

      expect(await screen.findByLabelText("gpt-4", {}, { timeout: 5000 })).toBeInTheDocument();
    });

    it("ignores a team the user has no access to", async () => {
      renderCreateKey({
        teams: [{ team_id: "team-1", models: [] } as unknown as Team],
        autoOpenCreate: true,
        prefillData: { team_id: "team-404", key_alias: "example-key" },
      });

      await userEvent.type(await screen.findByLabelText(/Key Name/), "-suffix");
      await submit();

      const payload = await createdPayload();
      expect(payload.team_id).toBeNull();
      expect(payload.key_alias).toBe("example-key-suffix");
    });

    it("falls back to you when another_user is requested by a non-admin", async () => {
      state.authorized = { ...state.authorized, userRole: "Internal User" };
      renderCreateKey({ autoOpenCreate: true, prefillData: { owned_by: "another_user", key_alias: "example-key" } });

      expect(await screen.findByRole("radio", { name: "You" })).toBeChecked();
    });

    it("applies another_user for an admin", async () => {
      renderCreateKey({ autoOpenCreate: true, prefillData: { owned_by: "another_user" } });

      expect(await screen.findByRole("radio", { name: "Another User" })).toBeChecked();
    });

    it("prefills the key type", async () => {
      renderCreateKey({ autoOpenCreate: true, prefillData: { key_type: "management" } });

      await screen.findByLabelText(/Key Name/);
      await userEvent.type(await screen.findByLabelText(/Key Name/), "prefilled-type");
      await submit();

      expect((await createdPayload()).key_type).toBe("management");
    });
  });

  describe("models dropdown team gating", () => {
    it("offers all-proxy-models but not all-team-models when no team is selected", async () => {
      await openModal();
      await userEvent.click(await screen.findByLabelText("Models"));

      expect(await screen.findByRole("option", { name: "All Proxy Models" })).toBeInTheDocument();
      expect(screen.queryByRole("option", { name: "All Team Models" })).not.toBeInTheDocument();
    });

    it("offers all-team-models but hides all-proxy-models once a team is selected", async () => {
      state.teams = [{ team_id: "team-1", team_alias: "Team One", models: ["team-model-1"] }];
      await openModal({ teams: state.teams as unknown as Team[] });

      await userEvent.click(await screen.findByLabelText("Team"));
      await userEvent.click(await screen.findByRole("option", { name: /Team One/ }));

      await userEvent.click(await screen.findByLabelText("Models"));

      expect(await screen.findByRole("option", { name: "All Team Models" })).toBeInTheDocument();
      expect(screen.queryByRole("option", { name: "All Proxy Models" })).not.toBeInTheDocument();
    });
  });

  describe("organization dropdown", () => {
    it("is editable for an admin", async () => {
      state.organizations = [{ organization_id: "org-1", organization_alias: "Engineering" }];
      await openModal();

      expect(await screen.findByLabelText("Organization")).not.toHaveAttribute("data-disabled");
    });

    it("is read-only for a non-admin", async () => {
      state.authorized = { ...state.authorized, userRole: "Internal User" };
      state.organizations = [{ organization_id: "org-1", organization_alias: "Engineering" }];
      await openModal();

      expect(await screen.findByLabelText("Organization")).toHaveAttribute("data-disabled", "");
    });

    it("routes a chosen organization into organization_id", async () => {
      state.organizations = [{ organization_id: "org-1", organization_alias: "Engineering" }];
      await openModal();
      await nameTheKey();

      await userEvent.click(await screen.findByLabelText("Organization"));
      await userEvent.click(await screen.findByRole("option", { name: /Engineering/ }));
      await submit();

      expect((await createdPayload()).organization_id).toBe("org-1");
    });
  });

  describe("policy and prompt fields", () => {
    it("loads and offers both selectors for a role that can see them", async () => {
      await openModal();
      await openSection(/Optional Settings/i);

      expect(await screen.findByLabelText("Policies")).toBeInTheDocument();
      expect(await screen.findByLabelText("Prompts")).toBeInTheDocument();
    });

    it("omits both selectors for a role that cannot see them", async () => {
      state.can = { viewPolicies: false, viewPrompts: false };
      await openModal();
      await openSection(/Optional Settings/i);

      await screen.findByLabelText("Guardrails");
      expect(screen.queryByLabelText("Policies")).not.toBeInTheDocument();
      expect(screen.queryByLabelText("Prompts")).not.toBeInTheDocument();
    });
  });

  describe("tags dropdown", () => {
    it("offers the tags returned by the tags hook", async () => {
      state.tags = { production: { name: "production" }, staging: { name: "staging" } };
      await openModal();
      await openSection(/Optional Settings/i);

      await userEvent.click(await screen.findByLabelText("Tags"));

      expect(await screen.findByTitle("production")).toBeInTheDocument();
      expect(await screen.findByTitle("staging")).toBeInTheDocument();
    });
  });

  describe("user search debounce", () => {
    it("fires exactly one search carrying the last typed value", async () => {
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      vi.useFakeTimers({ shouldAdvanceTime: true });
      try {
        renderCreateKey({ autoOpenCreate: true, prefillData: { owned_by: "another_user" } });
        const search = await userSearchInput();

        await user.type(search, "ali");
        expect(vi.mocked(userFilterUICall)).not.toHaveBeenCalled();

        await user.type(search, "ce");
        await vi.advanceTimersByTimeAsync(400);

        expect(vi.mocked(userFilterUICall)).toHaveBeenCalledTimes(1);
        const params = vi.mocked(userFilterUICall).mock.calls[0][1] as URLSearchParams;
        expect(params.get("user_email")).toBe("alice");
      } finally {
        vi.useRealTimers();
      }
    });

    it("keeps the current search's users when an abandoned search answers last", async () => {
      const answers = new Map<string, (users: { user_id: string; user_email: string }[]) => void>();
      vi.mocked(userFilterUICall).mockImplementation(
        (_accessToken, params) =>
          new Promise((resolve) => {
            answers.set(params.get("user_email") ?? "", resolve);
          }) as never,
      );

      const user = userEvent.setup();
      renderCreateKey({ autoOpenCreate: true, prefillData: { owned_by: "another_user" } });
      const search = await userSearchInput();

      await user.type(search, "ali");
      await waitFor(() => expect(answers.has("ali")).toBe(true), { timeout: 3000 });

      await user.type(search, "ce.smith@example.com");
      await waitFor(() => expect(answers.has("alice.smith@example.com")).toBe(true), { timeout: 3000 });

      await act(async () => {
        answers.get("alice.smith@example.com")?.([{ user_id: "u-smith", user_email: "alice.smith@example.com" }]);
      });
      await screen.findByTitle("alice.smith@example.com (u-smith)");

      await act(async () => {
        answers.get("ali")?.([{ user_id: "u-jones", user_email: "alice.jones@example.com" }]);
      });

      expect(screen.queryByTitle("alice.jones@example.com (u-jones)")).not.toBeInTheDocument();
      expect(screen.getByTitle("alice.smith@example.com (u-smith)")).toBeInTheDocument();
    });

    it("stops searching once the box is cleared and the abandoned search answers", async () => {
      const answers = new Map<string, (users: { user_id: string; user_email: string }[]) => void>();
      vi.mocked(userFilterUICall).mockImplementation(
        (_accessToken, params) =>
          new Promise((resolve) => {
            answers.set(params.get("user_email") ?? "", resolve);
          }) as never,
      );

      const user = userEvent.setup();
      renderCreateKey({ autoOpenCreate: true, prefillData: { owned_by: "another_user" } });
      const search = await userSearchInput();

      await user.type(search, "ali");
      await waitFor(() => expect(answers.has("ali")).toBe(true), { timeout: 3000 });
      await screen.findByText("Searching...");

      await user.clear(search);
      await screen.findByText("No users found");

      await act(async () => {
        answers.get("ali")?.([{ user_id: "u-jones", user_email: "alice.jones@example.com" }]);
      });

      expect(screen.queryByTitle("alice.jones@example.com (u-jones)")).not.toBeInTheDocument();
      expect(screen.getByText("No users found")).toBeInTheDocument();
    });

    it("keeps searching while a newer search is still in flight", async () => {
      const answers = new Map<string, (users: { user_id: string; user_email: string }[]) => void>();
      vi.mocked(userFilterUICall).mockImplementation(
        (_accessToken, params) =>
          new Promise((resolve) => {
            answers.set(params.get("user_email") ?? "", resolve);
          }) as never,
      );

      const user = userEvent.setup();
      renderCreateKey({ autoOpenCreate: true, prefillData: { owned_by: "another_user" } });
      const search = await userSearchInput();

      await user.type(search, "ali");
      await waitFor(() => expect(answers.has("ali")).toBe(true), { timeout: 3000 });

      await user.type(search, "ce.smith@example.com");
      await waitFor(() => expect(answers.has("alice.smith@example.com")).toBe(true), { timeout: 3000 });

      await act(async () => {
        answers.get("ali")?.([]);
      });

      expect(screen.getByText("Searching...")).toBeInTheDocument();
      expect(screen.queryByText("No users found")).not.toBeInTheDocument();

      await act(async () => {
        answers.get("alice.smith@example.com")?.([{ user_id: "u-smith", user_email: "alice.smith@example.com" }]);
      });
      await screen.findByTitle("alice.smith@example.com (u-smith)");
    });

    it("only warns about a failed search when it is the one the box is waiting on", async () => {
      const answers = new Map<
        string,
        { resolve: (users: { user_id: string; user_email: string }[]) => void; reject: (error: Error) => void }
      >();
      vi.mocked(userFilterUICall).mockImplementation(
        (_accessToken, params) =>
          new Promise((resolve, reject) => {
            answers.set(params.get("user_email") ?? "", { resolve, reject });
          }) as never,
      );

      const user = userEvent.setup();
      renderCreateKey({ autoOpenCreate: true, prefillData: { owned_by: "another_user" } });
      const search = await userSearchInput();

      await user.type(search, "ali");
      await waitFor(() => expect(answers.has("ali")).toBe(true), { timeout: 3000 });

      await user.type(search, "ce.smith@example.com");
      await waitFor(() => expect(answers.has("alice.smith@example.com")).toBe(true), { timeout: 3000 });

      await act(async () => {
        answers
          .get("alice.smith@example.com")
          ?.resolve([{ user_id: "u-smith", user_email: "alice.smith@example.com" }]);
      });
      await screen.findByTitle("alice.smith@example.com (u-smith)");

      await act(async () => {
        answers.get("ali")?.reject(new Error("search failed"));
      });

      expect(toast.fromError).not.toHaveBeenCalled();
      expect(screen.getByTitle("alice.smith@example.com (u-smith)")).toBeInTheDocument();

      await user.type(search, "x");
      await waitFor(() => expect(answers.has("alice.smith@example.comx")).toBe(true), { timeout: 3000 });

      await act(async () => {
        answers.get("alice.smith@example.comx")?.reject(new Error("search failed"));
      });

      expect(toast.fromError).toHaveBeenCalledTimes(1);
    });
  });

  describe("created key display", () => {
    it("surfaces the generated key after a successful create", async () => {
      await openModal();
      await nameTheKey();
      await submit();
      await createdPayload();

      expect(await screen.findByText("Save your Key")).toBeInTheDocument();
    });

    it("rejects a duplicate alias within the same team without calling the API", async () => {
      await openModal({ data: [{ team_id: null, key_alias: "contract-key" }] });
      await nameTheKey();
      await submit();

      await waitFor(() => {
        expect(screen.getByRole("button", { name: /^create key$/i })).toBeInTheDocument();
      });
      expect(vi.mocked(keyCreateCall)).not.toHaveBeenCalled();
    });
  });

  describe("dialog accessible names", () => {
    it("names the create form dialog", async () => {
      await openModal();

      expect(screen.getByRole("dialog", { name: "Create New Key" })).toBeInTheDocument();
    });

    it("names the created key dialog", async () => {
      await openModal();
      await nameTheKey();
      await submit();
      await createdPayload();

      expect(await screen.findByRole("dialog", { name: "Save your Key" })).toBeInTheDocument();
    });
  });

  describe("key type gating", () => {
    it("labels the llm_api option AI APIs", async () => {
      await openModal();
      await userEvent.click(await screen.findByLabelText("Key Type"));

      expect(
        await screen.findByText("Can call only AI API routes (chat/completions, embeddings, etc.)"),
      ).toBeInTheDocument();
      expect(screen.queryByText("LLM API")).not.toBeInTheDocument();
    });

    it("clears and disables models when a management key type is chosen", async () => {
      await openModal();
      await nameTheKey();

      await userEvent.click(await screen.findByLabelText("Key Type"));
      await userEvent.click(await screen.findByRole("option", { name: /^Management/ }));
      await submit();

      const payload = await createdPayload();
      expect(payload.key_type).toBe("management");
      expect(payload.models).toStrictEqual([]);
    });
  });

  describe("mount-gate liveness", () => {
    it("keeps the MCP tool permissions key out of the payload even with its section open", async () => {
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await openSection(SECTIONS.mcp);
      await submit();

      const payload = await createdPayload();
      expect(payload).not.toHaveProperty("mcp_tool_permissions");
      expect(payload).toHaveProperty("allowed_mcp_servers_and_groups");
    });

    it("registers no Optional Settings field while a team choice is still required", async () => {
      vi.mocked(modelAvailableCall).mockResolvedValue({ data: [{ id: "no-default-models" }] });
      renderCreateKey();
      await userEvent.click(screen.getByTestId("create-key-button"));

      expect(await screen.findByText(/Please select a team to continue/)).toBeInTheDocument();
      expect(screen.queryByLabelText(/Key Name/)).not.toBeInTheDocument();
      expect(screen.queryByRole("button", { name: /Optional Settings/i })).not.toBeInTheDocument();
    });
  });

  describe("writers outside the submit path", () => {
    it("lets the selected user win over the search text typed into the same field", async () => {
      vi.mocked(userFilterUICall).mockResolvedValue([
        { user_id: "u-77", user_email: "alice@example.com" },
      ] as unknown as Awaited<ReturnType<typeof userFilterUICall>>);

      await openModal();
      await userEvent.click(screen.getByRole("radio", { name: "Another User" }));
      await nameTheKey();

      await userEvent.type(await userSearchInput(), "alice");
      await userEvent.click(await screen.findByRole("option", { name: "alice@example.com (u-77)" }));
      await submit();

      expect((await createdPayload()).user_id).toBe("u-77");
    });

    it("surfaces the required message on a field that carries no help text", async () => {
      await openModal();
      await userEvent.click(screen.getByRole("radio", { name: "Another User" }));
      await nameTheKey();
      await submit();

      expect(
        await screen.findByText("Please input the user ID of the user you are assigning the key to"),
      ).toBeInTheDocument();
      expect(vi.mocked(keyCreateCall)).not.toHaveBeenCalled();
    });
  });

  describe("validation follows the mounted set", () => {
    it("submits an over-ceiling budget typed into a section the user closed again, omitting the key", async () => {
      await openModal({ team: { team_id: "team-1", max_budget: 10 } as unknown as Team });
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await userEvent.type(await screen.findByLabelText(/Max Budget \(USD\)/), "50");
      await openSection(/Optional Settings/i);
      await submit();

      const payload = await createdPayload();
      expect(payload).not.toHaveProperty("max_budget");
      expect(payload.key_alias).toBe("contract-key");
    });
  });

  describe("submit gestures", () => {
    it("creates the key when Enter is pressed inside a text field", async () => {
      await openModal();
      await userEvent.type(await screen.findByLabelText(/Key Name/), "enter-key{Enter}");

      expect((await createdPayload()).key_alias).toBe("enter-key");
    });
  });

  describe("switch coercion", () => {
    it("sends enable_prompt_caching as a boolean once the switch is on", async () => {
      await openModal();
      await nameTheKey();
      await openSection(/Optional Settings/i);
      await userEvent.click(await screen.findByRole("switch", { name: /Enable Prompt Caching/ }));
      await submit();

      expect((await createdPayload()).enable_prompt_caching).toBe(true);
    });
  });
});
