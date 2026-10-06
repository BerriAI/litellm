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

describe("CredentialModal with Anthropic workload identity federation", () => {
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

  it("refuses to save a federated credential that is missing its required ids", async () => {
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
});
