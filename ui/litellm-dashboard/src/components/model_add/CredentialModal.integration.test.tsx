import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { chooseSelectOption } from "../../../tests/test-utils";
import { Providers } from "../provider_info_helpers";
import { CredentialItem } from "../networking";
import CredentialModal from "./CredentialModal";

vi.mock("../networking", async () => {
  const actual = await vi.importActual("../networking");
  return {
    ...actual,
    getProviderCreateMetadata: vi.fn().mockResolvedValue([
      {
        provider: "OpenAI",
        provider_display_name: Providers.OpenAI,
        litellm_provider: "openai",
        credential_fields: [
          { key: "api_base", label: "API Base", field_type: "text", default_value: "https://api.openai.com/v1" },
          { key: "api_key", label: "OpenAI API Key", field_type: "password" },
        ],
      },
      {
        provider: "Anthropic",
        provider_display_name: Providers.Anthropic,
        litellm_provider: "anthropic",
        credential_fields: [
          { key: "api_base", label: "Upstream API Base", field_type: "text" },
          { key: "api_key", label: "API Key", field_type: "password" },
        ],
      },
      {
        provider: "Azure",
        provider_display_name: Providers.Azure,
        litellm_provider: "azure",
        credential_fields: [
          { key: "api_base", label: "Azure API Base", field_type: "text" },
          { key: "api_version", label: "API Version", field_type: "text" },
          { key: "api_key", label: "Azure API Key", field_type: "password" },
        ],
      },
      {
        provider: "MICROSOFT_365_COPILOT",
        provider_display_name: Providers.MICROSOFT_365_COPILOT,
        litellm_provider: "microsoft_365_copilot",
        credential_fields: [
          { key: "token_exchange_endpoint", label: "Token Endpoint URL", field_type: "text" },
          {
            key: "token_exchange_profile",
            label: "Exchange Grant",
            field_type: "select",
            options: ["jwt_bearer_obo", "rfc8693"],
            default_value: "jwt_bearer_obo",
          },
          { key: "client_id", label: "Client ID", field_type: "text" },
          { key: "client_secret", label: "Client Secret", field_type: "password" },
          { key: "token_exchange_scope", label: "Scope", field_type: "text" },
          { key: "token_exchange_audience", label: "Audience", field_type: "text" },
          { key: "api_key", label: "Delegated Access Token", field_type: "password" },
        ],
      },
    ]),
  };
});

const federatedCredential: CredentialItem = {
  credential_name: "anthropic-federated",
  credential_values: {
    anthropic_federation_rule_id: "fdrl_stored",
    anthropic_organization_id: "org-stored",
    anthropic_identity_source: "keycloak",
    anthropic_keycloak_token_url: "http****",
    anthropic_keycloak_client_id: "lite****",
    anthropic_keycloak_client_secret_ref: "os.e****",
  },
  credential_info: { custom_llm_provider: "anthropic" },
};

const azureCredential: CredentialItem = {
  credential_name: "azure-prod",
  credential_values: { api_base: "https://corp.openai.azure.com", api_version: "2024-10-21", api_key: "sk-1****" },
  credential_info: { custom_llm_provider: "Azure" },
};

const keyAndFederationCredential: CredentialItem = {
  credential_name: "anthropic-key-and-federation",
  credential_values: {
    api_key: "sk-a****",
    anthropic_federation_rule_id: "fdrl_stored",
    anthropic_organization_id: "org-stored",
    anthropic_identity_token_file: "/var/run/secrets/anthropic/token",
  },
  credential_info: { custom_llm_provider: "anthropic" },
};

const unknownSourceCredential: CredentialItem = {
  credential_name: "anthropic-unknown-source",
  credential_values: {
    anthropic_federation_rule_id: "fdrl_stored",
    anthropic_organization_id: "org-stored",
    anthropic_identity_source: "spiffe",
  },
  credential_info: { custom_llm_provider: "anthropic" },
};

const microsoftCopilotCredential: CredentialItem = {
  credential_name: "microsoft-copilot",
  credential_values: {
    token_exchange_endpoint: "https://identity.example.com/stored-token",
    token_exchange_profile: "jwt_bearer_obo",
    client_id: "stored-client",
    client_secret: "stored-secret",
    token_exchange_scope: "https://graph.microsoft.com/.default",
    token_exchange_audience: "https://graph.microsoft.com",
  },
  credential_info: { custom_llm_provider: "MICROSOFT_365_COPILOT" },
};

const renderModal = (props: Partial<React.ComponentProps<typeof CredentialModal>> = {}) => {
  const onSubmit = vi.fn();
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })}>
      <CredentialModal open mode="add" onCancel={vi.fn()} onSubmit={onSubmit} {...props} />
    </QueryClientProvider>,
  );
  return onSubmit;
};

const fill = (label: string | RegExp, value: string) =>
  fireEvent.change(screen.getByLabelText(label), { target: { value } });

const chooseOption = (user: ReturnType<typeof userEvent.setup>, select: RegExp, option: string | RegExp) =>
  chooseSelectOption(user, screen.getByRole("combobox", { name: select }), option);

const chooseProvider = async (user: ReturnType<typeof userEvent.setup>, provider: string) => {
  const providerSelect = screen.getByRole("combobox", { name: /Provider/ });
  await user.clear(providerSelect);
  await user.type(providerSelect, provider);
  const options = await screen.findAllByRole("option");
  await user.click(options.find((option) => option.textContent === provider) ?? options[0]);
};

describe("CredentialModal with Anthropic workload identity federation", () => {
  it("deletes stored exchange fields when switching to a static token", async () => {
    const user = userEvent.setup();
    const onSubmit = renderModal({ mode: "edit", existingCredential: microsoftCopilotCredential });

    expect(await screen.findByRole("combobox", { name: "Auth Type:" })).toHaveTextContent(
      "OAuth token exchange (on-behalf-of)",
    );
    await screen.findByLabelText("Token Endpoint URL");
    await chooseOption(user, /^Auth Type:/, "Static delegated access token");
    fill("Delegated Access Token", "delegated-graph-token");
    await user.click(screen.getByRole("button", { name: "Update Credential" }));

    const [values, valuesToDelete] = onSubmit.mock.calls[0];
    expect(values).toEqual({
      credential_name: "microsoft-copilot",
      custom_llm_provider: "MICROSOFT_365_COPILOT",
      api_key: "delegated-graph-token",
    });
    expect([...valuesToDelete].sort()).toEqual([
      "client_id",
      "client_secret",
      "token_exchange_audience",
      "token_exchange_endpoint",
      "token_exchange_profile",
      "token_exchange_scope",
    ]);
  });

  it("does not carry the previous provider's base URL into the Anthropic form or its payload", async () => {
    const user = userEvent.setup();
    const onSubmit = renderModal();
    expect(await screen.findByLabelText("API Base")).toHaveValue("https://api.openai.com/v1");

    const providerSelect = screen.getByRole("combobox", { name: /Provider/ });
    await user.click(providerSelect);
    await user.type(providerSelect, "Anthropic");
    await user.click((await screen.findAllByRole("option"))[0]);

    expect(await screen.findByLabelText("Upstream API Base")).toHaveValue("");
    fill("Credential Name:", "anthropic-key");
    await user.click(screen.getByRole("button", { name: "Add Credential" }));
    expect(onSubmit).toHaveBeenCalledWith({ credential_name: "anthropic-key", custom_llm_provider: "Anthropic" }, []);
  });

  it("creates a federated credential from the federation fields and never sends an API key", async () => {
    const user = userEvent.setup();
    const onSubmit = renderModal({ initialProvider: "Anthropic" });
    await screen.findByLabelText("API Key");
    fill("API Key", "sk-ant-typed-before-switching");

    await chooseOption(user, /^Authentication:/, "Workload identity federation");

    expect(screen.queryByLabelText("API Key")).not.toBeInTheDocument();
    fill("Credential Name:", "anthropic-federated");
    fill(/Federation Rule ID/, "fdrl_new");
    fill(/Organization ID/, "org-new");
    fill(/Identity Token File/, "/var/run/secrets/anthropic/token");
    await user.click(screen.getByRole("button", { name: "Add Credential" }));

    const expectedPayload = {
      credential_name: "anthropic-federated",
      custom_llm_provider: "Anthropic",
      anthropic_federation_rule_id: "fdrl_new",
      anthropic_organization_id: "org-new",
      anthropic_identity_token_file: "/var/run/secrets/anthropic/token",
    };
    expect(onSubmit).toHaveBeenCalledWith(expectedPayload, []);
  });

  it("names the identity source when the admin picks Keycloak", async () => {
    const user = userEvent.setup();
    const onSubmit = renderModal({ initialProvider: "Anthropic", initialAuthMethod: "federation" });
    await screen.findByLabelText("Upstream API Base");

    await chooseOption(user, /Identity Source/, "Keycloak client credentials");

    expect(screen.queryByLabelText(/Identity Token File/)).not.toBeInTheDocument();
    fill("Credential Name:", "anthropic-keycloak");
    fill(/Federation Rule ID/, "fdrl_new");
    fill(/Organization ID/, "org-new");
    fill(/Keycloak Token URL/, "https://keycloak.example.com/realms/litellm/protocol/openid-connect/token");
    fill(/Keycloak Client ID/, "litellm-proxy");
    fill(/Client Secret Reference/, "os.environ/KEYCLOAK_CLIENT_SECRET");
    await user.click(screen.getByRole("button", { name: "Add Credential" }));

    expect(onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        anthropic_identity_source: "keycloak",
        anthropic_keycloak_client_secret_ref: "os.environ/KEYCLOAK_CLIENT_SECRET",
      }),
      [],
    );
  });

  it("refuses to save a federated credential that is missing its identity token file", async () => {
    const user = userEvent.setup();
    const onSubmit = renderModal({ initialProvider: "Anthropic", initialAuthMethod: "federation" });
    await screen.findByLabelText("Upstream API Base");
    fill("Credential Name:", "anthropic-federated");
    fill(/Federation Rule ID/, "fdrl_new");

    await user.click(screen.getByRole("button", { name: "Add Credential" }));

    expect(await screen.findAllByText("Required")).not.toHaveLength(0);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("shows a stored federated credential's fields and writes nothing when it is saved untouched", async () => {
    const user = userEvent.setup();
    const onSubmit = renderModal({ mode: "edit", existingCredential: federatedCredential });

    expect(await screen.findByRole("combobox", { name: /^Authentication:/ })).toHaveTextContent(
      "Workload identity federation",
    );
    expect(screen.getByRole("combobox", { name: /Identity Source/ })).toHaveTextContent("Keycloak client credentials");
    expect(screen.getByLabelText(/Federation Rule ID/)).toHaveValue("fdrl_stored");
    expect(screen.getByLabelText(/Client Secret Reference/)).toHaveValue("os.e****");
    expect(screen.queryByLabelText("API Key")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Update Credential" }));

    expect(onSubmit).toHaveBeenCalledWith(
      { credential_name: "anthropic-federated", custom_llm_provider: "anthropic" },
      [],
    );
  });

  it("sends only the federation value the admin edited", async () => {
    const user = userEvent.setup();
    const onSubmit = renderModal({ mode: "edit", existingCredential: federatedCredential });
    await screen.findByLabelText("Upstream API Base");
    fill(/Organization ID/, "org-edited");

    await user.click(screen.getByRole("button", { name: "Update Credential" }));

    expect(onSubmit).toHaveBeenCalledWith(
      {
        credential_name: "anthropic-federated",
        custom_llm_provider: "anthropic",
        anthropic_organization_id: "org-edited",
      },
      [],
    );
  });

  it("deletes the stored federation values when the admin switches the credential to an API key", async () => {
    const user = userEvent.setup();
    const onSubmit = renderModal({ mode: "edit", existingCredential: federatedCredential });
    await screen.findByLabelText("Upstream API Base");

    await chooseOption(user, /^Authentication:/, "API key");
    await screen.findByLabelText("API Key");
    fill("API Key", "sk-ant-replacement");
    await user.click(screen.getByRole("button", { name: "Update Credential" }));

    const [values, valuesToDelete] = onSubmit.mock.calls[0];
    expect(values).toEqual({
      credential_name: "anthropic-federated",
      custom_llm_provider: "anthropic",
      api_key: "sk-ant-replacement",
    });
    expect([...valuesToDelete].sort()).toEqual(Object.keys(federatedCredential.credential_values).sort());
  });

  it("drops another provider's stored values when the admin turns the credential into a federated Anthropic one", async () => {
    const user = userEvent.setup();
    const onSubmit = renderModal({ mode: "edit", existingCredential: azureCredential });
    expect(await screen.findByLabelText("Azure API Base")).toHaveValue("https://corp.openai.azure.com");

    await chooseProvider(user, "Anthropic");
    expect(await screen.findByLabelText("Upstream API Base")).toHaveValue("");
    await chooseOption(user, /^Authentication:/, "Workload identity federation");
    fill(/Federation Rule ID/, "fdrl_new");
    fill(/Organization ID/, "org-new");
    fill(/Identity Token File/, "/var/run/secrets/anthropic/token");
    await user.click(screen.getByRole("button", { name: "Update Credential" }));

    const [values, valuesToDelete] = onSubmit.mock.calls[0];
    const expectedValues = {
      credential_name: "azure-prod",
      custom_llm_provider: "Anthropic",
      anthropic_federation_rule_id: "fdrl_new",
      anthropic_organization_id: "org-new",
      anthropic_identity_token_file: "/var/run/secrets/anthropic/token",
    };
    expect(values).toEqual(expectedValues);
    expect([...valuesToDelete].sort()).toEqual(["api_base", "api_key", "api_version"]);
  });

  it("shows the stored values again when the admin returns to the credential's own provider", async () => {
    const user = userEvent.setup();
    const onSubmit = renderModal({ mode: "edit", existingCredential: azureCredential });
    await screen.findByLabelText("Azure API Base");

    await chooseProvider(user, "Anthropic");
    await screen.findByLabelText("Upstream API Base");
    await chooseProvider(user, "Azure");

    expect(await screen.findByLabelText("Azure API Base")).toHaveValue("https://corp.openai.azure.com");
    await user.click(screen.getByRole("button", { name: "Update Credential" }));
    expect(onSubmit).toHaveBeenCalledWith({ credential_name: "azure-prod", custom_llm_provider: "Azure" }, []);
  });

  it("deletes the stored base URL when the admin clears it on a federated credential", async () => {
    const user = userEvent.setup();
    const stored = {
      ...federatedCredential,
      credential_values: { ...federatedCredential.credential_values, api_base: "https://gateway.example.com" },
    };
    const onSubmit = renderModal({ mode: "edit", existingCredential: stored });
    expect(await screen.findByLabelText("Upstream API Base")).toHaveValue("https://gateway.example.com");

    fill("Upstream API Base", "");
    await user.click(screen.getByRole("button", { name: "Update Credential" }));

    expect(onSubmit).toHaveBeenCalledWith(
      { credential_name: "anthropic-federated", custom_llm_provider: "anthropic" },
      ["api_base"],
    );
  });

  it("deletes the stored federation values when the admin moves a federated credential to another provider", async () => {
    const user = userEvent.setup();
    const onSubmit = renderModal({ mode: "edit", existingCredential: federatedCredential });
    await screen.findByLabelText(/Federation Rule ID/);

    await chooseProvider(user, "OpenAI");
    await screen.findByLabelText("OpenAI API Key");
    fill("OpenAI API Key", "sk-openai-new");
    await user.click(screen.getByRole("button", { name: "Update Credential" }));

    const [values, valuesToDelete] = onSubmit.mock.calls[0];
    expect(values).toMatchObject({ custom_llm_provider: "OpenAI", api_key: "sk-openai-new" });
    expect([...valuesToDelete].sort()).toEqual(Object.keys(federatedCredential.credential_values).sort());
  });

  it("opens a credential that stores an API key next to federation values as the API key credential the proxy uses", async () => {
    const user = userEvent.setup();
    const onSubmit = renderModal({ mode: "edit", existingCredential: keyAndFederationCredential });
    expect(await screen.findByRole("combobox", { name: /^Authentication:/ })).toHaveTextContent("API key");
    expect(await screen.findByLabelText("API Key")).toHaveValue("sk-a****");

    await chooseOption(user, /^Authentication:/, "Workload identity federation");
    expect(screen.getByLabelText(/Federation Rule ID/)).toHaveValue("fdrl_stored");
    await user.click(screen.getByRole("button", { name: "Update Credential" }));

    expect(onSubmit).toHaveBeenCalledWith(
      { credential_name: "anthropic-key-and-federation", custom_llm_provider: "anthropic" },
      ["api_key"],
    );
  });

  it("keeps what the admin typed in the form after submitting, so a failed save loses nothing", async () => {
    const user = userEvent.setup();
    const onSubmit = renderModal({ initialProvider: "Anthropic", initialAuthMethod: "federation" });
    await screen.findByLabelText("Upstream API Base");
    fill("Credential Name:", "anthropic-federated");
    fill(/Federation Rule ID/, "fdrl_new");
    fill(/Organization ID/, "org-new");
    fill(/Identity Token File/, "/var/run/secrets/anthropic/token");

    await user.click(screen.getByRole("button", { name: "Add Credential" }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(screen.getByLabelText("Credential Name:")).toHaveValue("anthropic-federated");
    expect(screen.getByLabelText(/Federation Rule ID/)).toHaveValue("fdrl_new");
    expect(screen.getByLabelText(/Identity Token File/)).toHaveValue("/var/run/secrets/anthropic/token");
  });

  it("refuses a required federation field that holds only whitespace", async () => {
    const user = userEvent.setup();
    const onSubmit = renderModal({ initialProvider: "Anthropic", initialAuthMethod: "federation" });
    await screen.findByLabelText("Upstream API Base");
    await chooseOption(user, /Identity Source/, "Token signed by LiteLLM (internal issuer)");
    fill("Credential Name:", "anthropic-issuer");
    fill(/Federation Rule ID/, "fdrl_new");
    fill(/Organization ID/, "org-new");
    fill(/Issuer URL/, "https://litellm.example.com");
    fill(/^Subject/, " ");
    fill(/Signing Key Reference/, "os.environ/ANTHROPIC_ISSUER_SIGNING_KEY");

    await user.click(screen.getByRole("button", { name: "Add Credential" }));

    expect(await screen.findByText("Required")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("shows an identity source it does not recognize as stored and leaves it alone on save", async () => {
    const user = userEvent.setup();
    const onSubmit = renderModal({ mode: "edit", existingCredential: unknownSourceCredential });

    const sourceSelect = await screen.findByRole("combobox", { name: /Identity Source/ });
    expect(sourceSelect).toHaveTextContent("spiffe");
    expect(sourceSelect).not.toHaveTextContent("Proxy environment variables");
    fill(/Organization ID/, "org-edited");
    await user.click(screen.getByRole("button", { name: "Update Credential" }));

    expect(onSubmit).toHaveBeenCalledWith(
      {
        credential_name: "anthropic-unknown-source",
        custom_llm_provider: "anthropic",
        anthropic_organization_id: "org-edited",
      },
      [],
    );
  });

  it("replaces an identity source it does not recognize when the admin picks one it offers", async () => {
    const user = userEvent.setup();
    const onSubmit = renderModal({ mode: "edit", existingCredential: unknownSourceCredential });
    await screen.findByRole("combobox", { name: /Identity Source/ });

    await chooseOption(user, /Identity Source/, "Proxy environment variables");
    await user.click(screen.getByRole("button", { name: "Update Credential" }));

    expect(onSubmit).toHaveBeenCalledWith(
      { credential_name: "anthropic-unknown-source", custom_llm_provider: "anthropic" },
      ["anthropic_identity_source"],
    );
  });

  it("refuses a proxy environment credential with no ids until the admin enters one", async () => {
    const user = userEvent.setup();
    const onSubmit = renderModal({ initialProvider: "Anthropic", initialAuthMethod: "federation" });
    await screen.findByLabelText("Upstream API Base");
    await chooseOption(user, /Identity Source/, "Proxy environment variables");
    fill("Credential Name:", "anthropic-env");

    await user.click(screen.getByRole("button", { name: "Add Credential" }));

    expect(await screen.findByText(/Enter at least one of these ids/)).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();

    fill(/Organization ID/, "org-new");
    await user.click(screen.getByRole("button", { name: "Add Credential" }));

    expect(onSubmit).toHaveBeenCalledWith(
      { credential_name: "anthropic-env", custom_llm_provider: "Anthropic", anthropic_organization_id: "org-new" },
      [],
    );
  });

  it("saves a federated credential that leaves the rule id and organization id to the proxy environment", async () => {
    const user = userEvent.setup();
    const onSubmit = renderModal({ initialProvider: "Anthropic", initialAuthMethod: "federation" });
    await screen.findByLabelText("Upstream API Base");
    fill("Credential Name:", "anthropic-env-ids");
    fill(/Identity Token File/, "/var/run/secrets/anthropic/token");

    await user.click(screen.getByRole("button", { name: "Add Credential" }));

    expect(onSubmit).toHaveBeenCalledWith(
      {
        credential_name: "anthropic-env-ids",
        custom_llm_provider: "Anthropic",
        anthropic_identity_token_file: "/var/run/secrets/anthropic/token",
      },
      [],
    );
  });
});
