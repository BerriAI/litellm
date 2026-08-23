import { renderWithProviders, screen, waitFor, within } from "../../../../../../tests/test-utils";
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
          ],
          variants: [{ id: "api_key", label: "API Key", field_keys: ["api_base", "api_key"], fixed_values: {} }],
        },
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
    expect(createProviderModelCall).toHaveBeenCalledWith("test-access-token", {
      model_name: "claude-3-opus",
      litellm_params: { model: "anthropic/claude-3-opus", litellm_credential_name: "anthropic-prod" },
      model_info: {},
      blocked: true,
    });
    expect(createProviderModelCall).toHaveBeenCalledWith("test-access-token", {
      model_name: "claude-3-haiku",
      litellm_params: { model: "anthropic/claude-3-haiku", litellm_credential_name: "anthropic-prod" },
      model_info: {},
      blocked: false,
    });
    expect(createProviderModelCall).toHaveBeenCalledWith("test-access-token", {
      model_name: "claude-hidden",
      litellm_params: { model: "anthropic/claude-hidden", litellm_credential_name: "anthropic-prod" },
      model_info: {},
      blocked: false,
    });

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
});
