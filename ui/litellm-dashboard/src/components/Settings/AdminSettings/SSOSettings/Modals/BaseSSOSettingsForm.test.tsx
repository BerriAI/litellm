import { Form } from "antd";
import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { renderWithProviders } from "../../../../../../tests/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import BaseSSOSettingsForm, { renderProviderFields, ssoProviderConfigs } from "./BaseSSOSettingsForm";

describe("BaseSSOSettingsForm", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("should render", () => {
    const TestWrapper = () => {
      const [form] = Form.useForm();
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
      const [form] = Form.useForm();
      const handleSubmit = vi.fn();

      return <BaseSSOSettingsForm form={form} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    const providerSelect = screen.getByLabelText("SSO Provider");
    await act(async () => {
      fireEvent.mouseDown(providerSelect);
    });

    await waitFor(() => {
      const googleOption = screen.getByText(/google sso/i);
      fireEvent.click(googleOption);
    });

    await waitFor(() => {
      expect(screen.getByText("Google Client ID")).toBeInTheDocument();
      expect(screen.getByText("Google Client Secret")).toBeInTheDocument();
    });
  });

  it("should show role mappings fields for okta provider", async () => {
    const TestWrapper = () => {
      const [form] = Form.useForm();
      const handleSubmit = vi.fn();

      return <BaseSSOSettingsForm form={form} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    const providerSelect = screen.getByLabelText("SSO Provider");
    await act(async () => {
      fireEvent.mouseDown(providerSelect);
    });

    await waitFor(() => {
      const oktaOption = screen.getByText(/okta/i);
      fireEvent.click(oktaOption);
    });

    await waitFor(() => {
      expect(screen.getByText("Use Role Mappings")).toBeInTheDocument();
    });
  });

  it("should validate proxy base url format", async () => {
    const TestWrapper = () => {
      const [form] = Form.useForm();
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
      const [form] = Form.useForm();
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
      const [form] = Form.useForm();
      const handleSubmit = vi.fn();

      return <BaseSSOSettingsForm form={form} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    const providerSelect = screen.getByLabelText("SSO Provider");
    await act(async () => {
      fireEvent.mouseDown(providerSelect);
    });

    await waitFor(() => {
      const genericOption = screen.getByText(/generic sso/i);
      fireEvent.click(genericOption);
    });

    await waitFor(() => {
      expect(screen.getByText("Use Role Mappings")).toBeInTheDocument();
    });

    const checkbox = screen.getByLabelText("Use Role Mappings");
    await act(async () => {
      fireEvent.click(checkbox);
    });

    await waitFor(() => {
      expect(screen.getByText("Group Claim")).toBeInTheDocument();
      expect(screen.getByText("Default Role")).toBeInTheDocument();
    });
  });

  it("should show team mappings checkbox for okta provider", async () => {
    const TestWrapper = () => {
      const [form] = Form.useForm();
      const handleSubmit = vi.fn();

      return <BaseSSOSettingsForm form={form} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    const providerSelect = screen.getByLabelText("SSO Provider");
    await act(async () => {
      fireEvent.mouseDown(providerSelect);
    });

    await waitFor(() => {
      const oktaOption = screen.getByText(/okta/i);
      fireEvent.click(oktaOption);
    });

    await waitFor(() => {
      expect(screen.getByText("Use Team Mappings")).toBeInTheDocument();
    });
  });

  it("should show team mappings checkbox for generic provider", async () => {
    const TestWrapper = () => {
      const [form] = Form.useForm();
      const handleSubmit = vi.fn();

      return <BaseSSOSettingsForm form={form} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    const providerSelect = screen.getByLabelText("SSO Provider");
    await act(async () => {
      fireEvent.mouseDown(providerSelect);
    });

    await waitFor(() => {
      const genericOption = screen.getByText(/generic sso/i);
      fireEvent.click(genericOption);
    });

    await waitFor(() => {
      expect(screen.getByText("Use Team Mappings")).toBeInTheDocument();
    });
  });

  it("should show team IDs JWT field when use_team_mappings is checked for okta provider", async () => {
    const TestWrapper = () => {
      const [form] = Form.useForm();
      const handleSubmit = vi.fn();

      return <BaseSSOSettingsForm form={form} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    const providerSelect = screen.getByLabelText("SSO Provider");
    await act(async () => {
      fireEvent.mouseDown(providerSelect);
    });

    await waitFor(() => {
      const oktaOption = screen.getByText(/okta/i);
      fireEvent.click(oktaOption);
    });

    await waitFor(() => {
      expect(screen.getByText("Use Team Mappings")).toBeInTheDocument();
    });

    const checkbox = screen.getByLabelText("Use Team Mappings");
    await act(async () => {
      fireEvent.click(checkbox);
    });

    await waitFor(() => {
      expect(screen.getByText("Team IDs JWT Field")).toBeInTheDocument();
    });
  });

  it("should not show team mappings checkbox for google provider", async () => {
    const TestWrapper = () => {
      const [form] = Form.useForm();
      const handleSubmit = vi.fn();

      return <BaseSSOSettingsForm form={form} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    const providerSelect = screen.getByLabelText("SSO Provider");
    await act(async () => {
      fireEvent.mouseDown(providerSelect);
    });

    await waitFor(() => {
      const googleOption = screen.getByText(/google sso/i);
      fireEvent.click(googleOption);
    });

    await waitFor(() => {
      expect(screen.getByText("Google Client ID")).toBeInTheDocument();
    });

    expect(screen.queryByText("Use Team Mappings")).not.toBeInTheDocument();
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
    let form: any;
    const TestWrapper = () => {
      const [formInstance] = Form.useForm();
      form = formInstance;
      return <BaseSSOSettingsForm form={formInstance} onFormSubmit={handleSubmit} />;
    };

    renderWithProviders(<TestWrapper />);

    // Mirror EditSSOSettingsModal hydrating the form from the GET response.
    await act(async () => {
      form.setFieldsValue({
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
      form.setFieldsValue({ generic_token_endpoint: "https://idp.example.com/token/v2" });
      form.submit();
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
      const [form] = Form.useForm();
      return <BaseSSOSettingsForm form={form} onFormSubmit={vi.fn()} />;
    };

    renderWithProviders(<TestWrapper />);

    await act(async () => {
      fireEvent.mouseDown(screen.getByLabelText("SSO Provider"));
    });

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
});
