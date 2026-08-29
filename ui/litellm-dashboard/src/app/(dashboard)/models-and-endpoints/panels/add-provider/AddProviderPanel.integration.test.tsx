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
            { key: "anthropic_issuer_url", label: "Issuer URL", field_type: "text", required: true },
            { key: "anthropic_issuer_subject", label: "Issuer Subject", field_type: "text", required: true },
            {
              key: "anthropic_issuer_signing_key_ref",
              label: "Signing Key Reference",
              field_type: "text",
              required: true,
            },
          ],
          variants: [
            { id: "api_key", label: "API Key", field_keys: ["api_base", "api_key"], fixed_values: {} },
            {
              id: "wif_internal_issuer",
              label: "Workload Identity Federation (LiteLLM-signed)",
              field_keys: [
                "anthropic_federation_rule_id",
                "anthropic_organization_id",
                "anthropic_issuer_url",
                "anthropic_issuer_subject",
                "anthropic_issuer_signing_key_ref",
              ],
              optional_field_keys: ["anthropic_federation_rule_id"],
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
  renderWithProviders(<AddProviderPanel />);
  await screen.findByLabelText("Provider");
  return { user };
};

const chooseProvider = async (user: ReturnType<typeof userEvent.setup>, name: string) => {
  await user.click(screen.getByLabelText("Provider"));
  await user.click(await screen.findByText(name));
};

const rowFor = (upstreamId: string) => within(screen.getByText(upstreamId).closest("tr") as HTMLElement);

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
    const { user } = await setup();

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

  it("saves a LiteLLM-signed credential with a blank federation rule id, then PATCHes it from the JWKS step", async () => {
    discoverProviderModelsCall.mockResolvedValue({ models: ["claude-3-opus"] });
    const { user } = await setup();

    await chooseProvider(user, "Anthropic");
    await user.type(screen.getByLabelText("Credential name"), "anthropic-wif");
    await user.click(screen.getByRole("button", { name: /Next/ }));

    await chooseSelectOption(
      user,
      await screen.findByRole("combobox", { name: "Authentication method" }),
      "Workload Identity Federation (LiteLLM-signed)",
    );

    fireEvent.change(await screen.findByLabelText("Organization ID"), { target: { value: "org-1" } });
    fireEvent.change(screen.getByLabelText("Issuer URL"), { target: { value: "https://proxy.example.com" } });
    fireEvent.change(screen.getByLabelText("Issuer Subject"), { target: { value: "litellm-proxy" } });
    fireEvent.change(screen.getByLabelText("Signing Key Reference"), { target: { value: "os.environ/SIGNING_KEY" } });
    expect(screen.getByLabelText("Federation Rule ID")).toHaveValue("");

    await user.click(screen.getByRole("button", { name: "Save credential" }));

    // The rule id is only readable off the Anthropic Console once the JWKS below is registered,
    // and the JWKS only exists once the credential is saved, so saving must not demand it first.
    await waitFor(() =>
      expect(credentialCreateCall).toHaveBeenCalledWith("test-access-token", {
        credential_name: "anthropic-wif",
        credential_values: {
          anthropic_organization_id: "org-1",
          anthropic_issuer_url: "https://proxy.example.com",
          anthropic_issuer_subject: "litellm-proxy",
          anthropic_issuer_signing_key_ref: "os.environ/SIGNING_KEY",
          anthropic_identity_source: "internal_issuer",
        },
        credential_info: { custom_llm_provider: "anthropic" },
      }),
    );

    expect(await screen.findByText("Register this JWKS with Anthropic")).toBeInTheDocument();
    expect(getCredentialJwksCall).toHaveBeenCalledWith("test-access-token", "anthropic-wif");

    fireEvent.change(screen.getByLabelText("Federation Rule ID"), { target: { value: "rule-abc" } });
    await user.click(screen.getByRole("button", { name: /Next/ }));

    await waitFor(() =>
      expect(credentialUpdateCall).toHaveBeenCalledWith("test-access-token", "anthropic-wif", {
        credential_name: "anthropic-wif",
        credential_values: { anthropic_federation_rule_id: "rule-abc" },
        credential_info: { custom_llm_provider: "anthropic" },
      }),
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
    fireEvent.change(await screen.findByLabelText("Organization ID"), { target: { value: "org-1" } });
    fireEvent.change(screen.getByLabelText("Issuer URL"), { target: { value: "https://proxy.example.com" } });
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
    fireEvent.change(await screen.findByLabelText("Organization ID"), { target: { value: "org-1" } });
    fireEvent.change(screen.getByLabelText("Issuer URL"), { target: { value: "https://proxy.example.com" } });
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
