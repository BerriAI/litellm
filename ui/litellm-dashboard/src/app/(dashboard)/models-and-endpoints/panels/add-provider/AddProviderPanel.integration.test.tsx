import {
  chooseSelectOption,
  fireEvent,
  renderWithProviders,
  screen,
  waitFor,
  within,
} from "../../../../../../tests/test-utils";
import userEvent, { PointerEventsCheckLevel } from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AddProviderPanel from "./AddProviderPanel";

const discoverProviderModelsCall = vi.fn();
const credentialCreateCall = vi.fn();
const credentialUpdateCall = vi.fn();
const createProviderModelCall = vi.fn();
const listAllModelsCall = vi.fn();
const getCallbacksCall = vi.fn();
const setCallbacksCall = vi.fn();
const getCredentialJwksCall = vi.fn();
const mockAuthorized = vi.fn();

vi.mock("@/components/networking", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/components/networking")>();
  return {
    ...actual,
    discoverProviderModelsCall: (...args: unknown[]) => discoverProviderModelsCall(...args),
    credentialCreateCall: (...args: unknown[]) => credentialCreateCall(...args),
    credentialUpdateCall: (...args: unknown[]) => credentialUpdateCall(...args),
    createProviderModelCall: (...args: unknown[]) => createProviderModelCall(...args),
    listAllModelsCall: (...args: unknown[]) => listAllModelsCall(...args),
    getCallbacksCall: (...args: unknown[]) => getCallbacksCall(...args),
    setCallbacksCall: (...args: unknown[]) => setCallbacksCall(...args),
    getCredentialJwksCall: (...args: unknown[]) => getCredentialJwksCall(...args),
  };
});

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({ default: () => mockAuthorized() }));

vi.mock("@/app/(dashboard)/hooks/credentials/useCredentials", () => ({
  useCredentials: () => ({ data: { credentials: [] } }),
}));

vi.mock("@/app/(dashboard)/hooks/providers/useProviderFields", () => ({
  useProviderFields: () => ({
    data: [
      {
        provider: "Anthropic",
        provider_display_name: "Anthropic",
        litellm_provider: "anthropic",
        default_model_placeholder: "claude-3-opus",
        credential_fields: [
          { key: "api_base", label: "Upstream API Base", field_type: "text" },
          { key: "api_key", label: "API Key", field_type: "password" },
        ],
        credential_variants: {
          selector_label: "Authentication method",
          default_variant: "api_key",
          field_definitions: [
            { key: "api_base", label: "Upstream API Base", field_type: "text" },
            { key: "api_key", label: "API Key", field_type: "password" },
            {
              key: "anthropic_federation_rule_id",
              label: "Federation Rule ID",
              field_type: "text",
              required: true,
              tooltip: "Can be left blank and filled in once the JWKS is registered.",
            },
            { key: "anthropic_organization_id", label: "Organization ID", field_type: "text", required: true },
            { key: "anthropic_service_account_id", label: "Service Account ID", field_type: "text", required: false },
            { key: "anthropic_workspace_id", label: "Workspace ID", field_type: "text", required: false },
            { key: "anthropic_issuer_url", label: "Issuer URL", field_type: "text", required: true },
            { key: "anthropic_issuer_subject", label: "Issuer Subject", field_type: "text", required: true },
            {
              key: "anthropic_issuer_signing_key_ref",
              label: "Signing Key Reference",
              field_type: "text",
              required: true,
            },
            { key: "anthropic_identity_token", label: "Identity Token Reference", field_type: "text", required: true },
          ],
          variants: [
            { id: "api_key", label: "API Key", field_keys: ["api_base", "api_key"], fixed_values: {} },
            {
              id: "wif_token",
              label: "Workload Identity Federation (external token)",
              field_keys: [
                "anthropic_federation_rule_id",
                "anthropic_organization_id",
                "anthropic_service_account_id",
                "anthropic_workspace_id",
                "anthropic_identity_token",
              ],
              fixed_values: {},
            },
            {
              id: "wif_internal_issuer",
              label: "Workload Identity Federation (LiteLLM-signed)",
              field_keys: [
                "anthropic_issuer_url",
                "anthropic_issuer_subject",
                "anthropic_issuer_signing_key_ref",
                "anthropic_organization_id",
                "anthropic_federation_rule_id",
                "anthropic_service_account_id",
                "anthropic_workspace_id",
              ],
              optional_field_keys: ["anthropic_organization_id", "anthropic_federation_rule_id"],
              fixed_values: { anthropic_identity_source: "internal_issuer" },
            },
          ],
        },
      },
      {
        provider: "OpenAI",
        provider_display_name: "OpenAI",
        litellm_provider: "openai",
        default_model_placeholder: "gpt-4o",
        credential_fields: [{ key: "api_key", label: "API Key", field_type: "password" }],
      },
    ],
    isLoading: false,
    error: null,
  }),
}));

const PROXY_ADMIN = { accessToken: "test-access-token" };

const setup = async () => {
  const user = userEvent.setup({ pointerEventsCheck: PointerEventsCheckLevel.Never });
  const onClose = vi.fn();
  renderWithProviders(<AddProviderPanel onClose={onClose} />);
  await screen.findByLabelText("Provider");
  return { user, onClose };
};

const chooseProvider = async (user: ReturnType<typeof userEvent.setup>, name: string) => {
  await user.click(screen.getByLabelText("Provider"));
  await user.click(await screen.findByText(name));
};

const rowFor = (upstreamId: string) =>
  within(screen.getByRole("row", { name: (accessibleName) => accessibleName.startsWith(`${upstreamId} `) }));

const INTERNAL_ISSUER_CREATE_VALUES = {
  anthropic_issuer_url: "https://proxy.example.com",
  anthropic_issuer_subject: "litellm-proxy",
  anthropic_issuer_signing_key_ref: "os.environ/SIGNING_KEY",
  anthropic_identity_source: "internal_issuer",
};

const saveInternalIssuerCredential = async (user: ReturnType<typeof userEvent.setup>, name: string) => {
  await chooseProvider(user, "Anthropic");
  await user.type(screen.getByLabelText("Credential name"), name);
  await user.click(screen.getByRole("button", { name: /Next/ }));
  await chooseSelectOption(
    user,
    await screen.findByRole("combobox", { name: "Authentication method" }),
    "Workload Identity Federation (LiteLLM-signed)",
  );
  fireEvent.change(await screen.findByLabelText("Issuer URL"), { target: { value: "https://proxy.example.com" } });
  fireEvent.change(screen.getByLabelText("Issuer Subject"), { target: { value: "litellm-proxy" } });
  fireEvent.change(screen.getByLabelText("Signing Key Reference"), { target: { value: "os.environ/SIGNING_KEY" } });
  await user.click(screen.getByRole("button", { name: "Save credential" }));
  await screen.findByText("Register this JWKS with Anthropic");
};

const fillFederationIds = (ids: Record<string, string>) => {
  for (const [label, value] of Object.entries(ids)) {
    fireEvent.change(screen.getByLabelText(label), { target: { value } });
  }
};

const FEDERATION_ID_LABELS = ["Organization ID", "Federation Rule ID", "Service Account ID", "Workspace ID"] as const;

const expectNoFederationIdFields = () => {
  for (const label of FEDERATION_ID_LABELS) {
    expect(screen.queryByLabelText(label)).not.toBeInTheDocument();
  }
};

describe("AddProviderPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockAuthorized.mockReturnValue(PROXY_ADMIN);
    credentialCreateCall.mockResolvedValue({});
    credentialUpdateCall.mockResolvedValue({});
    listAllModelsCall.mockResolvedValue({ data: [] });
    createProviderModelCall.mockResolvedValue({ model_id: "new-id" });
    getCallbacksCall.mockResolvedValue({ router_settings: {} });
    setCallbacksCall.mockResolvedValue({});
    getCredentialJwksCall.mockResolvedValue({ keys: [{ kid: "kid-1", kty: "RSA", n: "n", e: "AQAB" }] });
  });

  it("walks provider -> credential -> discover -> review -> create, with blocked and aliases wired correctly", async () => {
    discoverProviderModelsCall.mockResolvedValue({ models: ["claude-3-opus", "claude-3-haiku"] });
    const { user, onClose } = await setup();

    await chooseProvider(user, "Anthropic");
    await user.type(screen.getByLabelText("Credential name"), "anthropic-prod");
    await user.click(screen.getByRole("button", { name: /Next/ }));

    await user.type(await screen.findByLabelText("API Key"), "sk-ant-test");
    await user.click(screen.getByRole("button", { name: "Save credential" }));

    expect(await screen.findByText("claude-3-opus")).toBeInTheDocument();
    expect(credentialCreateCall).toHaveBeenCalledWith("test-access-token", {
      credential_name: "anthropic-prod",
      credential_values: { api_key: "sk-ant-test" },
      credential_info: { custom_llm_provider: "anthropic" },
    });
    expect(discoverProviderModelsCall).toHaveBeenCalledWith("test-access-token", {
      custom_llm_provider: "anthropic",
      litellm_credential_name: "anthropic-prod",
    });

    // Disable the first discovered row.
    await user.click(rowFor("claude-3-opus").getByRole("switch"));

    // Add an alternate name to the second discovered row.
    const haikuAltNames = rowFor("claude-3-haiku").getByRole("combobox");
    await user.type(haikuAltNames, "gpt-4o-mini");
    await user.click(await screen.findByText('Create "gpt-4o-mini"'));

    // Add a manual (hidden) model.
    await user.type(screen.getByPlaceholderText("upstream model id"), "claude-hidden");
    await user.click(screen.getByRole("button", { name: "Add" }));

    await user.click(screen.getByRole("button", { name: "Create 3 models" }));

    await waitFor(() => expect(createProviderModelCall).toHaveBeenCalledTimes(3));
    const disabledOpusCreation = {
      model_name: "claude-3-opus",
      litellm_params: { model: "anthropic/claude-3-opus", litellm_credential_name: "anthropic-prod" },
      model_info: {},
      blocked: true,
    };
    const renamedHaikuCreation = {
      model_name: "claude-3-haiku",
      litellm_params: { model: "anthropic/claude-3-haiku", litellm_credential_name: "anthropic-prod" },
      model_info: {},
      blocked: false,
    };
    const manualHiddenCreation = {
      model_name: "claude-hidden",
      litellm_params: { model: "anthropic/claude-hidden", litellm_credential_name: "anthropic-prod" },
      model_info: {},
      blocked: false,
    };
    expect(createProviderModelCall).toHaveBeenCalledWith("test-access-token", disabledOpusCreation);
    expect(createProviderModelCall).toHaveBeenCalledWith("test-access-token", renamedHaikuCreation);
    expect(createProviderModelCall).toHaveBeenCalledWith("test-access-token", manualHiddenCreation);

    await waitFor(() =>
      expect(setCallbacksCall).toHaveBeenCalledWith("test-access-token", {
        router_settings: { model_group_alias: { "gpt-4o-mini": "claude-3-haiku" } },
      }),
    );

    expect(await screen.findByText(/claude-3-opus: created/)).toBeInTheDocument();
    expect(screen.getByText(/claude-3-haiku: created/)).toBeInTheDocument();
    expect(screen.getByText(/claude-hidden: created/)).toBeInTheDocument();

    expect(onClose).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Close" }));
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("shows a sanitized discovery error with a working retry", async () => {
    discoverProviderModelsCall.mockRejectedValueOnce(new Error("upstream auth failed"));
    discoverProviderModelsCall.mockResolvedValueOnce({ models: ["claude-3-opus"] });
    const { user } = await setup();

    await chooseProvider(user, "Anthropic");
    await user.type(screen.getByLabelText("Credential name"), "anthropic-prod");
    await user.click(screen.getByRole("button", { name: /Next/ }));
    await user.type(await screen.findByLabelText("API Key"), "sk-ant-test");
    await user.click(screen.getByRole("button", { name: "Save credential" }));

    expect(await screen.findByText("Discovery failed")).toBeInTheDocument();
    expect(screen.getByText("upstream auth failed")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Retry" }));

    expect(await screen.findByText("claude-3-opus")).toBeInTheDocument();
    expect(discoverProviderModelsCall).toHaveBeenCalledTimes(2);
  });

  it("skips a row already created under this credential on a re-run", async () => {
    discoverProviderModelsCall.mockResolvedValue({ models: ["claude-3-opus"] });
    listAllModelsCall.mockResolvedValue({
      data: [
        {
          model_name: "claude-3-opus",
          litellm_params: { model: "anthropic/claude-3-opus", litellm_credential_name: "anthropic-prod" },
          model_info: { id: "existing-id" },
        },
      ],
    });
    const { user } = await setup();

    await chooseProvider(user, "Anthropic");
    await user.type(screen.getByLabelText("Credential name"), "anthropic-prod");
    await user.click(screen.getByRole("button", { name: /Next/ }));
    await user.type(await screen.findByLabelText("API Key"), "sk-ant-test");
    await user.click(screen.getByRole("button", { name: "Save credential" }));

    await screen.findByText("claude-3-opus");
    await user.click(screen.getByRole("button", { name: "Create 1 model" }));

    expect(await screen.findByText(/claude-3-opus: skipped/)).toBeInTheDocument();
    expect(createProviderModelCall).not.toHaveBeenCalled();
  });

  it("does not persist an alias for a row whose model creation failed", async () => {
    discoverProviderModelsCall.mockResolvedValue({ models: ["claude-3-opus", "claude-3-haiku"] });
    createProviderModelCall.mockImplementation(async (_token: string, payload: { model_name: string }) => {
      if (payload.model_name === "claude-3-haiku") {
        throw new Error("upstream rejected");
      }
      return { model_id: "new-id" };
    });
    const { user } = await setup();

    await chooseProvider(user, "Anthropic");
    await user.type(screen.getByLabelText("Credential name"), "anthropic-prod");
    await user.click(screen.getByRole("button", { name: /Next/ }));
    await user.type(await screen.findByLabelText("API Key"), "sk-ant-test");
    await user.click(screen.getByRole("button", { name: "Save credential" }));

    expect(await screen.findByText("claude-3-opus")).toBeInTheDocument();

    const opusAltNames = rowFor("claude-3-opus").getByRole("combobox");
    await user.type(opusAltNames, "gpt-4o");
    await user.click(await screen.findByText('Create "gpt-4o"'));

    const haikuAltNames = rowFor("claude-3-haiku").getByRole("combobox");
    await user.type(haikuAltNames, "gpt-4o-mini");
    await user.click(await screen.findByText('Create "gpt-4o-mini"'));

    await user.click(screen.getByRole("button", { name: "Create 2 models" }));

    expect(await screen.findByText(/claude-3-opus: created/)).toBeInTheDocument();
    expect(await screen.findByText(/claude-3-haiku: failed/)).toBeInTheDocument();

    await waitFor(() =>
      expect(setCallbacksCall).toHaveBeenCalledWith("test-access-token", {
        router_settings: { model_group_alias: { "gpt-4o": "claude-3-opus" } },
      }),
    );
  });

  it("asks for the federation ids on the Register issuer step only, never on Authentication, for the LiteLLM-signed method", async () => {
    const { user } = await setup();

    await chooseProvider(user, "Anthropic");
    await user.type(screen.getByLabelText("Credential name"), "anthropic-wif");
    await user.click(screen.getByRole("button", { name: /Next/ }));

    // An external token means the rule already exists, so its ids are ordinary credential fields.
    await chooseSelectOption(
      user,
      await screen.findByRole("combobox", { name: "Authentication method" }),
      "Workload Identity Federation (external token)",
    );
    expect(await screen.findByLabelText("Identity Token Reference")).toBeInTheDocument();
    for (const label of FEDERATION_ID_LABELS) {
      expect(screen.getByLabelText(label)).toBeInTheDocument();
    }

    // Anthropic only issues the ids once the JWKS from the next step is registered, so asking for
    // them here would be asking for values the operator cannot have yet.
    await chooseSelectOption(
      user,
      screen.getByRole("combobox", { name: "Authentication method" }),
      "Workload Identity Federation (LiteLLM-signed)",
    );
    expect(await screen.findByLabelText("Issuer URL")).toBeInTheDocument();
    expectNoFederationIdFields();

    fireEvent.change(screen.getByLabelText("Issuer URL"), { target: { value: "https://proxy.example.com" } });
    fireEvent.change(screen.getByLabelText("Issuer Subject"), { target: { value: "litellm-proxy" } });
    fireEvent.change(screen.getByLabelText("Signing Key Reference"), { target: { value: "os.environ/SIGNING_KEY" } });
    await user.click(screen.getByRole("button", { name: "Save credential" }));
    expect(await screen.findByText("Register this JWKS with Anthropic")).toBeInTheDocument();
    for (const label of FEDERATION_ID_LABELS) {
      expect(screen.getByLabelText(label)).toHaveValue("");
    }
  });

  it("saves a LiteLLM-signed credential before any Anthropic id exists, then collects them all on the JWKS step", async () => {
    discoverProviderModelsCall.mockResolvedValue({ models: ["claude-3-opus"] });
    const { user } = await setup();

    await saveInternalIssuerCredential(user, "anthropic-wif");

    // Every id comes off the Anthropic Console only once the JWKS below is registered, and the
    // JWKS only exists once the credential is saved, so saving must not demand any of them first.
    expect(credentialCreateCall).toHaveBeenCalledWith("test-access-token", {
      credential_name: "anthropic-wif",
      credential_values: INTERNAL_ISSUER_CREATE_VALUES,
      credential_info: { custom_llm_provider: "anthropic" },
    });
    expect(getCredentialJwksCall).toHaveBeenCalledWith("test-access-token", "anthropic-wif");

    expect(screen.getByText("Still needed before discovery: Organization ID, Federation Rule ID.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Next/ })).toBeDisabled();

    fillFederationIds({ "Organization ID": "org-1" });
    expect(screen.getByText("Still needed before discovery: Federation Rule ID.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Next/ })).toBeDisabled();

    fillFederationIds({ "Federation Rule ID": " fdrl_abc ", "Service Account ID": "svac_1" });
    expect(screen.queryByText(/Still needed before discovery/)).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /Next/ }));

    await waitFor(() =>
      expect(credentialUpdateCall).toHaveBeenCalledWith("test-access-token", "anthropic-wif", {
        credential_name: "anthropic-wif",
        credential_values: {
          anthropic_organization_id: "org-1",
          anthropic_federation_rule_id: "fdrl_abc",
          anthropic_service_account_id: "svac_1",
        },
        credential_info: { custom_llm_provider: "anthropic" },
      }),
    );
    expect(credentialUpdateCall).toHaveBeenCalledTimes(1);
    expect(await screen.findByText("claude-3-opus")).toBeInTheDocument();
    expect(discoverProviderModelsCall).toHaveBeenCalledTimes(1);
  });

  it("keeps the ids from the JWKS step across Back, a credential re-save and the return trip", async () => {
    discoverProviderModelsCall.mockRejectedValueOnce(new Error("Authentication failed"));
    discoverProviderModelsCall.mockResolvedValueOnce({ models: ["claude-3-opus"] });
    const { user } = await setup();

    await saveInternalIssuerCredential(user, "anthropic-wif");
    fillFederationIds({ "Organization ID": "org-1", "Federation Rule ID": "fdrl_abc", "Service Account ID": "svac_1" });
    await user.click(screen.getByRole("button", { name: /Next/ }));
    expect(await screen.findByText("Discovery failed")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Back/ }));
    expect(await screen.findByText("Register this JWKS with Anthropic")).toBeInTheDocument();
    expect(screen.getByLabelText("Federation Rule ID")).toHaveValue("fdrl_abc");
    expect(screen.getByLabelText("Service Account ID")).toHaveValue("svac_1");

    await user.click(screen.getByRole("button", { name: /Back/ }));
    expect(await screen.findByLabelText("Issuer URL")).toHaveValue("https://proxy.example.com");
    expectNoFederationIdFields();

    // Re-saving Authentication must neither resend nor delete the ids it no longer mounts.
    credentialUpdateCall.mockClear();
    fireEvent.change(screen.getByLabelText("Issuer Subject"), { target: { value: "litellm-proxy-2" } });
    await user.click(screen.getByRole("button", { name: "Save changes" }));
    expect(await screen.findByText("Register this JWKS with Anthropic")).toBeInTheDocument();
    expect(credentialUpdateCall).toHaveBeenCalledWith("test-access-token", "anthropic-wif", {
      credential_name: "anthropic-wif",
      credential_values: { ...INTERNAL_ISSUER_CREATE_VALUES, anthropic_issuer_subject: "litellm-proxy-2" },
      credential_info: { custom_llm_provider: "anthropic" },
    });
    expect(screen.getByLabelText("Organization ID")).toHaveValue("org-1");
    expect(screen.getByLabelText("Federation Rule ID")).toHaveValue("fdrl_abc");
    expect(screen.getByLabelText("Service Account ID")).toHaveValue("svac_1");
    expect(screen.queryByText(/Still needed before discovery/)).not.toBeInTheDocument();

    credentialUpdateCall.mockClear();
    await user.click(screen.getByRole("button", { name: /Next/ }));
    expect(await screen.findByText("claude-3-opus")).toBeInTheDocument();
    expect(credentialUpdateCall).not.toHaveBeenCalled();
  });

  it("lets a failed discovery be fixed by adding the Workspace ID on the JWKS step, PATCHing only that id", async () => {
    discoverProviderModelsCall.mockRejectedValueOnce(new Error("Model discovery failed: HTTP 401"));
    discoverProviderModelsCall.mockResolvedValueOnce({ models: ["claude-3-opus"] });
    const { user } = await setup();

    await saveInternalIssuerCredential(user, "anthropic-wif");
    fillFederationIds({ "Organization ID": "org-1", "Federation Rule ID": "fdrl_abc" });
    await user.click(screen.getByRole("button", { name: /Next/ }));
    expect(await screen.findByText("Model discovery failed: HTTP 401")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Back/ }));
    fillFederationIds({ "Workspace ID": "wrkspc_1" });
    credentialUpdateCall.mockClear();
    await user.click(screen.getByRole("button", { name: /Next/ }));

    await waitFor(() =>
      expect(credentialUpdateCall).toHaveBeenCalledWith("test-access-token", "anthropic-wif", {
        credential_name: "anthropic-wif",
        credential_values: { anthropic_workspace_id: "wrkspc_1" },
        credential_info: { custom_llm_provider: "anthropic" },
      }),
    );
    expect(await screen.findByText("claude-3-opus")).toBeInTheDocument();
    expect(discoverProviderModelsCall).toHaveBeenCalledTimes(2);
  });

  it("deletes an id cleared on the JWKS step instead of leaving the saved value in place", async () => {
    discoverProviderModelsCall.mockRejectedValueOnce(new Error("Model discovery failed: HTTP 401"));
    discoverProviderModelsCall.mockResolvedValueOnce({ models: ["claude-3-opus"] });
    const { user } = await setup();

    await saveInternalIssuerCredential(user, "anthropic-wif");
    fillFederationIds({ "Organization ID": "org-1", "Federation Rule ID": "fdrl_abc", "Workspace ID": "wrkspc_stale" });
    await user.click(screen.getByRole("button", { name: /Next/ }));
    expect(await screen.findByText("Model discovery failed: HTTP 401")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Back/ }));
    expect(await screen.findByLabelText("Workspace ID")).toHaveValue("wrkspc_stale");

    fillFederationIds({ "Workspace ID": "" });
    await user.click(screen.getByRole("button", { name: /Next/ }));

    const workspaceDeletion = {
      credential_name: "anthropic-wif",
      credential_values: {},
      credential_info: { custom_llm_provider: "anthropic" },
      credential_values_to_delete: ["anthropic_workspace_id"],
    };
    await waitFor(() =>
      expect(credentialUpdateCall).toHaveBeenCalledWith("test-access-token", "anthropic-wif", workspaceDeletion),
    );
    expect(await screen.findByText("claude-3-opus")).toBeInTheDocument();
  });
  it("refuses to create when the deployment lookup fails, rather than duplicating saved rows", async () => {
    discoverProviderModelsCall.mockResolvedValue({ models: ["claude-3-opus"] });
    listAllModelsCall.mockRejectedValue(new Error("proxy unreachable"));
    const { user } = await setup();

    await chooseProvider(user, "Anthropic");
    await user.type(screen.getByLabelText("Credential name"), "anthropic-prod");
    await user.click(screen.getByRole("button", { name: /Next/ }));
    await user.type(await screen.findByLabelText("API Key"), "sk-ant-test");
    await user.click(screen.getByRole("button", { name: "Save credential" }));

    await screen.findByText("claude-3-opus");
    await user.click(screen.getByRole("button", { name: "Create 1 model" }));

    expect(await screen.findByText(/could duplicate ones already saved/i)).toBeInTheDocument();
    expect(createProviderModelCall).not.toHaveBeenCalled();
  });

  it("creates rather than PATCHes after the credential name changes following a save", async () => {
    discoverProviderModelsCall.mockResolvedValue({ models: ["claude-3-opus"] });
    const { user } = await setup();

    await chooseProvider(user, "Anthropic");
    await user.type(screen.getByLabelText("Credential name"), "anthropic-first");
    await user.click(screen.getByRole("button", { name: /Next/ }));

    await chooseSelectOption(
      user,
      await screen.findByRole("combobox", { name: "Authentication method" }),
      "Workload Identity Federation (LiteLLM-signed)",
    );
    fireEvent.change(await screen.findByLabelText("Issuer URL"), { target: { value: "https://proxy.example.com" } });
    fireEvent.change(screen.getByLabelText("Issuer Subject"), { target: { value: "litellm-proxy" } });
    fireEvent.change(screen.getByLabelText("Signing Key Reference"), { target: { value: "os.environ/SIGNING_KEY" } });
    await user.click(screen.getByRole("button", { name: "Save credential" }));

    // The JWKS step is the one place the wizard pauses after a save, so it is the route back to
    // the name field. Renaming there must create the new credential, never PATCH the old name.
    await screen.findByText("Register this JWKS with Anthropic");
    await user.click(screen.getByRole("button", { name: /Back/ }));
    await user.click(await screen.findByRole("button", { name: /Back/ }));

    const nameInput = await screen.findByLabelText("Credential name");
    await user.clear(nameInput);
    await user.type(nameInput, "anthropic-second");
    await user.click(screen.getByRole("button", { name: /Next/ }));

    credentialCreateCall.mockClear();
    credentialUpdateCall.mockClear();
    await user.click(await screen.findByRole("button", { name: /Save credential/ }));

    await waitFor(() =>
      expect(credentialCreateCall).toHaveBeenCalledWith(
        "test-access-token",
        expect.objectContaining({ credential_name: "anthropic-second" }),
      ),
    );
    expect(credentialUpdateCall).not.toHaveBeenCalled();
  });

  it("creates rather than PATCHes when the provider changes under the same credential name", async () => {
    discoverProviderModelsCall.mockResolvedValue({ models: ["claude-3-opus"] });
    const { user } = await setup();

    await chooseProvider(user, "Anthropic");
    await user.type(screen.getByLabelText("Credential name"), "shared-name");
    await user.click(screen.getByRole("button", { name: /Next/ }));

    await chooseSelectOption(
      user,
      await screen.findByRole("combobox", { name: "Authentication method" }),
      "Workload Identity Federation (LiteLLM-signed)",
    );
    fireEvent.change(await screen.findByLabelText("Issuer URL"), { target: { value: "https://proxy.example.com" } });
    fireEvent.change(screen.getByLabelText("Issuer Subject"), { target: { value: "litellm-proxy" } });
    fireEvent.change(screen.getByLabelText("Signing Key Reference"), { target: { value: "os.environ/SIGNING_KEY" } });
    await user.click(screen.getByRole("button", { name: "Save credential" }));

    // A credential is (name, provider). Keeping the name but switching provider must not PATCH the
    // Anthropic credential into an OpenAI one.
    await screen.findByText("Register this JWKS with Anthropic");
    await user.click(screen.getByRole("button", { name: /Back/ }));
    await user.click(await screen.findByRole("button", { name: /Back/ }));
    await chooseProvider(user, "OpenAI");
    await user.click(screen.getByRole("button", { name: /Next/ }));

    credentialCreateCall.mockClear();
    credentialUpdateCall.mockClear();
    await user.type(await screen.findByLabelText("API Key"), "sk-openai-test");
    await user.click(screen.getByRole("button", { name: /Save credential/ }));

    await waitFor(() =>
      expect(credentialCreateCall).toHaveBeenCalledWith(
        "test-access-token",
        expect.objectContaining({
          credential_name: "shared-name",
          credential_info: { custom_llm_provider: "openai" },
        }),
      ),
    );
    expect(credentialUpdateCall).not.toHaveBeenCalled();
  });

  it("drops the previous provider's credential values when the provider changes", async () => {
    discoverProviderModelsCall.mockResolvedValue({ models: ["gpt-4o"] });
    const { user } = await setup();

    await chooseProvider(user, "Anthropic");
    await user.type(screen.getByLabelText("Credential name"), "switching");
    await user.click(screen.getByRole("button", { name: /Next/ }));
    await user.type(await screen.findByLabelText("API Key"), "sk-ant-secret");
    await user.click(screen.getByRole("button", { name: /Back/ }));

    await chooseProvider(user, "OpenAI");
    await user.click(screen.getByRole("button", { name: /Next/ }));

    const apiKeyField = await screen.findByLabelText("API Key");
    expect(apiKeyField).toHaveValue("");
    await user.type(apiKeyField, "sk-openai-test");
    await user.click(screen.getByRole("button", { name: /Save credential/ }));

    await waitFor(() =>
      expect(credentialCreateCall).toHaveBeenCalledWith(
        "test-access-token",
        expect.objectContaining({
          credential_values: expect.objectContaining({ api_key: "sk-openai-test" }),
          credential_info: { custom_llm_provider: "openai" },
        }),
      ),
    );
  });

  it("blocks creation while any model name is blank", async () => {
    discoverProviderModelsCall.mockResolvedValue({ models: ["claude-3-opus"] });
    const { user } = await setup();

    await chooseProvider(user, "Anthropic");
    await user.type(screen.getByLabelText("Credential name"), "anthropic-prod");
    await user.click(screen.getByRole("button", { name: /Next/ }));
    await user.type(await screen.findByLabelText("API Key"), "sk-ant-test");
    await user.click(screen.getByRole("button", { name: "Save credential" }));

    const nameCell = await screen.findByDisplayValue("claude-3-opus");
    await user.clear(nameCell);

    expect(screen.getByRole("button", { name: /Create 1 model/ })).toBeDisabled();
  });
});
