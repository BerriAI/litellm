import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "../../../../../../tests/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import BaseSSOSettingsForm, {
  buildSSOSettingsPatch,
  emptySSOSettingsFormValues,
  pickMountedSSOValues,
  renderProviderFields,
  ssoProviderConfigs,
  submitMountedSSOValues,
  useSSOSettingsForm,
} from "./BaseSSOSettingsForm";
import { expectControlBesideLabel } from "../../../../../../tests/fieldOrientation";
import { processSSOSettingsPayload } from "../utils";

const user = () => userEvent.setup({ pointerEventsCheck: 0 });

const openProviderDropdown = async () => {
  await user().click(screen.getByLabelText("SSO Provider"));
};

describe("BaseSSOSettingsForm", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render", () => {
    const TestWrapper = () => {
      const form = useSSOSettingsForm("sso-settings");
      const handleSubmit = vi.fn();

      return <BaseSSOSettingsForm form={form} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    expect(screen.getByText("SSO Provider")).toBeInTheDocument();
    expect(screen.getByText("Proxy Admin Email")).toBeInTheDocument();
    expect(screen.getByText("Proxy Base URL")).toBeInTheDocument();
  });

  it("should render provider fields when provider is selected", async () => {
    const TestWrapper = () => {
      const form = useSSOSettingsForm("sso-settings");
      const handleSubmit = vi.fn();

      return <BaseSSOSettingsForm form={form} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    await openProviderDropdown();

    const googleOption = await screen.findByText(/google sso/i);
    await user().click(googleOption);

    await waitFor(() => {
      expect(screen.getByText("Google Client ID")).toBeInTheDocument();
      expect(screen.getByText("Google Client Secret")).toBeInTheDocument();
    });
  });

  it("should show role mappings fields for okta provider", async () => {
    const TestWrapper = () => {
      const form = useSSOSettingsForm("sso-settings");
      const handleSubmit = vi.fn();

      return <BaseSSOSettingsForm form={form} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    await openProviderDropdown();

    const oktaOption = await screen.findByText(/okta/i);
    await user().click(oktaOption);

    await waitFor(() => {
      expect(screen.getByText("Use Role Mappings")).toBeInTheDocument();
    });
  });

  it("should validate proxy base url format", async () => {
    const TestWrapper = () => {
      const form = useSSOSettingsForm("sso-settings");
      const handleSubmit = vi.fn();

      return <BaseSSOSettingsForm form={form} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    const urlInput = screen.getByPlaceholderText("https://example.com");
    await act(async () => {
      fireEvent.change(urlInput, { target: { value: "invalid-url" } });
      fireEvent.blur(urlInput);
    });

    await waitFor(() => {
      expect(screen.getByText(/URL must start with http:\/\/ or https:\/\//i)).toBeInTheDocument();
    });
  });

  it("should validate proxy base url trailing slash", async () => {
    const TestWrapper = () => {
      const form = useSSOSettingsForm("sso-settings");
      const handleSubmit = vi.fn();

      return <BaseSSOSettingsForm form={form} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    const urlInput = screen.getByPlaceholderText("https://example.com");
    await act(async () => {
      fireEvent.change(urlInput, { target: { value: "https://example.com/" } });
      fireEvent.blur(urlInput);
    });

    await waitFor(() => {
      expect(screen.getByText(/URL must not end with a trailing slash/i)).toBeInTheDocument();
    });
  });

  it("should show role mappings fields when use_role_mappings is checked for generic provider", async () => {
    const TestWrapper = () => {
      const form = useSSOSettingsForm("sso-settings");
      const handleSubmit = vi.fn();

      return <BaseSSOSettingsForm form={form} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    await openProviderDropdown();

    const genericOption = await screen.findByText(/generic sso/i);
    await user().click(genericOption);

    await waitFor(() => {
      expect(screen.getByText("Use Role Mappings")).toBeInTheDocument();
    });

    await user().click(screen.getAllByLabelText("Use Role Mappings")[0]);

    await waitFor(() => {
      expect(screen.getByText("Group Claim")).toBeInTheDocument();
      expect(screen.getByText("Default Role")).toBeInTheDocument();
    });
  });

  it("should show team mappings checkbox for okta provider", async () => {
    const TestWrapper = () => {
      const form = useSSOSettingsForm("sso-settings");
      const handleSubmit = vi.fn();

      return <BaseSSOSettingsForm form={form} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    await openProviderDropdown();

    const oktaOption = await screen.findByText(/okta/i);
    await user().click(oktaOption);

    await waitFor(() => {
      expect(screen.getByText("Use Team Mappings")).toBeInTheDocument();
    });
  });

  it("should show team mappings checkbox for generic provider", async () => {
    const TestWrapper = () => {
      const form = useSSOSettingsForm("sso-settings");
      const handleSubmit = vi.fn();

      return <BaseSSOSettingsForm form={form} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    await openProviderDropdown();

    const genericOption = await screen.findByText(/generic sso/i);
    await user().click(genericOption);

    await waitFor(() => {
      expect(screen.getByText("Use Team Mappings")).toBeInTheDocument();
    });
  });

  it("should show team IDs JWT field when use_team_mappings is checked for okta provider", async () => {
    const TestWrapper = () => {
      const form = useSSOSettingsForm("sso-settings");
      const handleSubmit = vi.fn();

      return <BaseSSOSettingsForm form={form} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    await openProviderDropdown();

    const oktaOption = await screen.findByText(/okta/i);
    await user().click(oktaOption);

    await waitFor(() => {
      expect(screen.getByText("Use Team Mappings")).toBeInTheDocument();
    });

    await user().click(screen.getAllByLabelText("Use Team Mappings")[0]);

    await waitFor(() => {
      expect(screen.getByText("Team IDs JWT Field")).toBeInTheDocument();
    });
  });

  it("should not show team mappings checkbox for google provider", async () => {
    const TestWrapper = () => {
      const form = useSSOSettingsForm("sso-settings");
      const handleSubmit = vi.fn();

      return <BaseSSOSettingsForm form={form} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    await openProviderDropdown();

    const googleOption = await screen.findByText(/google sso/i);
    await user().click(googleOption);

    await waitFor(() => {
      expect(screen.getByText("Google Client ID")).toBeInTheDocument();
    });

    expect(screen.queryByText("Use Team Mappings")).not.toBeInTheDocument();
  });

  it("lays a provider checkbox field out beside its label", async () => {
    const TestWrapper = () => {
      const form = useSSOSettingsForm("sso-settings");

      return <BaseSSOSettingsForm form={form} onFormSubmit={vi.fn()} />;
    };

    renderWithProviders(<TestWrapper />);

    await openProviderDropdown();
    await user().click(await screen.findByText(/saml sso/i));

    expectControlBesideLabel(
      await screen.findByRole("checkbox", { name: "Allow IdP-initiated (unsolicited) responses" }),
    );
  });

  it.each(["Use Role Mappings", "Use Team Mappings"])("lays the %s toggle out beside its label", async (label) => {
    const TestWrapper = () => {
      const form = useSSOSettingsForm("sso-settings");

      return <BaseSSOSettingsForm form={form} onFormSubmit={vi.fn()} />;
    };

    renderWithProviders(<TestWrapper />);

    await openProviderDropdown();
    await user().click(await screen.findByText(/okta/i));

    expectControlBesideLabel(await screen.findByRole("checkbox", { name: label }));
  });
});

describe("renderProviderFields", () => {
  it("should return null for unknown provider", () => {
    const result = renderProviderFields("unknown");
    expect(result).toBeNull();
  });

  it("should return fields for google provider", () => {
    const result = renderProviderFields("google");
    expect(result).not.toBeNull();
    expect(result?.length).toBe(2);
  });

  it("should return fields for microsoft provider", () => {
    const result = renderProviderFields("microsoft");
    expect(result).not.toBeNull();
    expect(result?.length).toBe(3);
  });

  it("should return fields for okta provider", () => {
    const result = renderProviderFields("okta");
    expect(result).not.toBeNull();
    expect(result?.length).toBe(6);
  });

  it("should return fields for generic provider", () => {
    const result = renderProviderFields("generic");
    expect(result).not.toBeNull();
    expect(result?.length).toBe(6);
  });

  it.each(["okta", "generic"])(
    "renders an optional generic_scope field for %s so editing cannot clear it",
    (provider) => {
      const scopeField = ssoProviderConfigs[provider].fields.find((field) => field.name === "generic_scope");
      expect(scopeField).toBeDefined();
      expect(scopeField?.required).toBe(false);
      expect(ssoProviderConfigs[provider].envVarMap.generic_scope).toBe("GENERIC_SCOPE");
    },
  );

  it("submits generic_scope untouched, so saving an unrelated edit cannot clear GENERIC_SCOPE", async () => {
    // update_sso_settings clears the env var for any mapped field its payload
    // omits, and antd only submits mounted fields. So the Scopes field being
    // present is what stops an unrelated edit from downgrading a custom scope
    // to the provider default. Dropping the field from ssoProviderConfigs must
    // fail here rather than silently in production.
    const handleSubmit = vi.fn();
    let form!: ReturnType<typeof useSSOSettingsForm>;
    const TestWrapper = () => {
      const formInstance = useSSOSettingsForm("sso-settings");
      form = formInstance;
      return <BaseSSOSettingsForm form={formInstance} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    // Mirror EditSSOSettingsModal hydrating the form from the GET response.
    await act(async () => {
      form.reset({
        ...emptySSOSettingsFormValues,
        sso_provider: "generic",
        generic_client_id: "client-id",
        generic_client_secret: "client-secret",
        generic_authorization_endpoint: "https://idp.example.com/authorize",
        generic_token_endpoint: "https://idp.example.com/token",
        generic_userinfo_endpoint: "https://idp.example.com/userinfo",
        generic_scope: "openid email profile groups",
        proxy_base_url: "https://gateway.example.com",
        user_email: "admin@example.com",
      });
    });

    // The admin edits something else entirely and saves.
    await act(async () => {
      form.setValue("generic_token_endpoint", "https://idp.example.com/token/v2");
      submitMountedSSOValues(form, "sso-settings", handleSubmit)();
    });

    await waitFor(() => {
      expect(handleSubmit).toHaveBeenCalledWith(
        expect.objectContaining({
          generic_token_endpoint: "https://idp.example.com/token/v2",
          generic_scope: "openid email profile groups",
        }),
      );
    });
  });

  it("renders provider logos in the dropdown and falls back to a letter avatar on load error", async () => {
    const TestWrapper = () => {
      const form = useSSOSettingsForm("sso-settings");
      return <BaseSSOSettingsForm form={form} onFormSubmit={vi.fn()} />;
    };

    renderWithProviders(<TestWrapper />);

    await openProviderDropdown();

    await waitFor(() => {
      expect(screen.getAllByAltText("Google SSO logo").length).toBeGreaterThan(0);
    });

    expect(screen.getAllByAltText("Google SSO logo")[0]).toHaveAttribute("src", expect.stringContaining("google.svg"));
    expect(screen.getAllByAltText("Microsoft SSO logo")[0]).toHaveAttribute(
      "src",
      expect.stringContaining("microsoft_azure.svg"),
    );
    expect(screen.queryByAltText("Generic SSO logo")).not.toBeInTheDocument();

    const oktaLogo = screen.getAllByAltText("Okta / Auth0 SSO logo")[0];
    expect(oktaLogo).toHaveAttribute("src", expect.stringContaining("https://www.okta.com/"));

    await act(async () => {
      fireEvent.error(oktaLogo);
    });

    await waitFor(() => {
      expect(screen.queryByAltText("Okta / Auth0 SSO logo")).not.toBeInTheDocument();
      expect(screen.getByText("O")).toBeInTheDocument();
    });
  });

  it("blocks the submit and names the missing provider credential", async () => {
    const handleSubmit = vi.fn();
    let form!: ReturnType<typeof useSSOSettingsForm>;
    const TestWrapper = () => {
      const formInstance = useSSOSettingsForm("sso-settings");
      form = formInstance;
      return <BaseSSOSettingsForm form={formInstance} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    await act(async () => {
      form.reset({
        ...emptySSOSettingsFormValues,
        sso_provider: "google",
        google_client_secret: "a-secret",
        user_email: "admin@example.com",
        proxy_base_url: "https://gateway.example.com",
      });
    });

    await act(async () => {
      submitMountedSSOValues(form, "sso-settings", handleSubmit)();
    });

    expect(await screen.findByText("Please enter the google client id")).toBeInTheDocument();
    expect(handleSubmit).not.toHaveBeenCalled();
  });

  it("shows the effective default role on an untouched form", async () => {
    let form!: ReturnType<typeof useSSOSettingsForm>;
    const TestWrapper = () => {
      const formInstance = useSSOSettingsForm("sso-settings");
      form = formInstance;
      return <BaseSSOSettingsForm form={formInstance} onFormSubmit={vi.fn()} />;
    };

    renderWithProviders(<TestWrapper />);

    await act(async () => {
      form.reset({ ...emptySSOSettingsFormValues, sso_provider: "okta", use_role_mappings: true });
    });

    expect(await screen.findByLabelText("Default Role")).toHaveTextContent("Internal User");
  });
});

describe("buildSSOSettingsPatch", () => {
  const seeded = {
    ...emptySSOSettingsFormValues,
    sso_provider: "generic",
    generic_client_id: "client-id",
    generic_client_secret: "kc-s****6e0d",
    generic_authorization_endpoint: "https://idp.example.com/authorize",
    generic_token_endpoint: "https://idp.example.com/token",
    generic_userinfo_endpoint: "https://idp.example.com/userinfo",
    generic_scope: "openid email profile groups",
    saml_idp_metadata_url: "https://idp.example.com/saml/metadata",
    saml_sp_entity_id: "litellm-sp",
    microsoft_client_id: "ms-client-id",
    microsoft_tenant: "ms-tenant",
    user_email: "admin@example.com",
    proxy_base_url: "https://gateway.example.com",
    use_role_mappings: true,
    group_claim: "groups",
    default_role: "internal_user",
    proxy_admin_teams: "admins",
  };
  const submit = (values: typeof seeded) => pickMountedSSOValues(values, "sso-settings");
  const patch = (submitted: typeof seeded) =>
    buildSSOSettingsPatch(seeded, submit(submitted), "sso-settings", processSSOSettingsPayload);

  it("sends only the edited field, leaving SAML and another provider's settings the form never showed", () => {
    expect(patch({ ...seeded, user_email: "new@example.com" })).toEqual({ user_email: "new@example.com" });
  });

  it("sends null for a shown field the admin emptied", () => {
    expect(patch({ ...seeded, generic_scope: "" })).toEqual({ generic_scope: null });
  });

  it("clears role mappings when the admin unchecks them", () => {
    expect(patch({ ...seeded, use_role_mappings: false })).toEqual({ role_mappings: null });
  });

  it("clears the previous provider's credentials when the admin switches provider", () => {
    const google = { sso_provider: "google", google_client_id: "g-id", google_client_secret: "g-secret" };
    const clearedGeneric = {
      generic_client_id: null,
      generic_client_secret: null,
      generic_authorization_endpoint: null,
      generic_token_endpoint: null,
      generic_userinfo_endpoint: null,
      generic_scope: null,
    };
    const clearedOthers = {
      microsoft_client_id: null,
      microsoft_tenant: null,
      saml_idp_metadata_url: null,
      saml_sp_entity_id: null,
    };
    expect(patch({ ...seeded, ...google })).toEqual({ ...google, ...clearedGeneric, ...clearedOthers });
  });

  it("clears a stored provider that would outrank the new one at sign-in, not only the one the form showed", () => {
    const googleAndMicrosoft = {
      ...emptySSOSettingsFormValues,
      sso_provider: "google",
      google_client_id: "g-id",
      microsoft_client_id: "ms-id",
      user_email: "admin@example.com",
      proxy_base_url: "https://gateway.example.com",
    };
    const generic = {
      ...googleAndMicrosoft,
      sso_provider: "generic",
      generic_client_id: "client-id",
      generic_client_secret: "secret",
      generic_authorization_endpoint: "https://idp.example.com/authorize",
      generic_token_endpoint: "https://idp.example.com/token",
      generic_userinfo_endpoint: "https://idp.example.com/userinfo",
    };
    const result = buildSSOSettingsPatch(
      googleAndMicrosoft,
      pickMountedSSOValues(generic, "sso-settings"),
      "sso-settings",
      processSSOSettingsPayload,
    );
    expect(result).toMatchObject({ google_client_id: null, microsoft_client_id: null, generic_client_id: "client-id" });
  });

  it("keeps the shared generic credentials when the admin switches between okta and generic", () => {
    const result = patch({ ...seeded, sso_provider: "okta" });
    expect(result.sso_provider).toBe("okta");
    expect(Object.keys(result).filter((key) => key.startsWith("generic_"))).toEqual([]);
  });
});
