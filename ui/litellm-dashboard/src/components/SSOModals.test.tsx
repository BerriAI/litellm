import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import SSOModals from "./SSOModals";
import { useSSOSettingsForm } from "./Settings/AdminSettings/SSOSettings/Modals/BaseSSOSettingsForm";

const user = () => userEvent.setup({ pointerEventsCheck: 0 });

// Mock the networking functions
vi.mock("./networking", () => ({
  getSSOSettings: vi.fn(),
  updateSSOSettings: vi.fn(),
}));

// Mock parseErrorMessage
vi.mock("./shared/errorUtils", () => ({
  parseErrorMessage: vi.fn((error) => error?.message || "An error occurred"),
}));

import { toast } from "@/lib/toast";
import { getSSOSettings, updateSSOSettings } from "./networking";

describe("SSOModals", () => {
  it("should render the SSOModals component", () => {
    const TestWrapper = () => {
      const form = useSSOSettingsForm("admin-panel");

      return (
        <SSOModals
          isAddSSOModalVisible={true}
          isInstructionsModalVisible={false}
          handleAddSSOOk={() => {}}
          handleAddSSOCancel={() => {}}
          handleShowInstructions={() => {}}
          handleInstructionsOk={() => {}}
          handleInstructionsCancel={() => {}}
          form={form}
          accessToken={null}
          ssoConfigured={false}
        />
      );
    };

    render(<TestWrapper />);
    expect(screen.getByText("Add SSO")).toBeInTheDocument();
  });

  it("should show validation error if proxy base url is not a valid URL", async () => {
    const TestWrapper = () => {
      const form = useSSOSettingsForm("admin-panel");
      return (
        <SSOModals
          isAddSSOModalVisible={true}
          isInstructionsModalVisible={false}
          handleAddSSOOk={() => {}}
          handleAddSSOCancel={() => {}}
          handleShowInstructions={() => {}}
          handleInstructionsOk={() => {}}
          handleInstructionsCancel={() => {}}
          form={form}
          accessToken={null}
          ssoConfigured={false}
        />
      );
    };

    render(<TestWrapper />);

    // Find and interact with the SSO provider select
    await user().click(screen.getByLabelText("SSO Provider"));
    // Wait for dropdown and select Google
    const googleOption = await screen.findByText("Google SSO");
    await user().click(googleOption);

    // Fill in the email field
    const emailInput = screen.getByLabelText("Proxy Admin Email");
    fireEvent.change(emailInput, { target: { value: "test@example.com" } });

    // Fill in an invalid URL
    const urlInput = screen.getByLabelText("Proxy Base URL");
    fireEvent.change(urlInput, { target: { value: "invalid-url" } });

    // Submit the form
    const saveButton = screen.getByText("Save");
    fireEvent.click(saveButton);

    // Check for validation error
    await waitFor(
      () => {
        expect(screen.getByText("URL must start with http:// or https://")).toBeInTheDocument();
      },
      // The validation is based on a Promise, so we need to wait for it to resolve
      { timeout: 5000 },
    );
  });

  it("should show validation error if proxy base url ends with trailing slash", async () => {
    const TestWrapper = () => {
      const form = useSSOSettingsForm("admin-panel");
      return (
        <SSOModals
          isAddSSOModalVisible={true}
          isInstructionsModalVisible={false}
          handleAddSSOOk={() => {}}
          handleAddSSOCancel={() => {}}
          handleShowInstructions={() => {}}
          handleInstructionsOk={() => {}}
          handleInstructionsCancel={() => {}}
          form={form}
          accessToken={null}
          ssoConfigured={false}
        />
      );
    };

    render(<TestWrapper />);

    // Find and interact with the SSO provider select
    await user().click(screen.getByLabelText("SSO Provider"));
    // Wait for dropdown and select Google
    const googleOption = await screen.findByText("Google SSO");
    await user().click(googleOption);

    // Fill in the email field
    const emailInput = screen.getByLabelText("Proxy Admin Email");
    fireEvent.change(emailInput, { target: { value: "test@example.com" } });

    // Fill in a URL with trailing slash
    const urlInput = screen.getByLabelText("Proxy Base URL") as HTMLInputElement;
    fireEvent.change(urlInput, { target: { value: "https://example.com/" } });

    // Submit the form
    const saveButton = screen.getByText("Save");
    fireEvent.click(saveButton);

    // Check for validation error using findByText for async rendering
    const errorMessage = await screen.findByText("URL must not end with a trailing slash", {}, { timeout: 5000 });
    expect(errorMessage).toBeInTheDocument();
  });

  it("should allow typing https:// without interfering with slashes", async () => {
    const TestWrapper = () => {
      const form = useSSOSettingsForm("admin-panel");
      return (
        <SSOModals
          isAddSSOModalVisible={true}
          isInstructionsModalVisible={false}
          handleAddSSOOk={() => {}}
          handleAddSSOCancel={() => {}}
          handleShowInstructions={() => {}}
          handleInstructionsOk={() => {}}
          handleInstructionsCancel={() => {}}
          form={form}
          accessToken={null}
          ssoConfigured={false}
        />
      );
    };

    render(<TestWrapper />);

    const urlInput = screen.getByLabelText("Proxy Base URL") as HTMLInputElement;

    // Simulate user typing "https://"
    fireEvent.change(urlInput, { target: { value: "h" } });
    expect(urlInput.value).toBe("h");

    fireEvent.change(urlInput, { target: { value: "ht" } });
    expect(urlInput.value).toBe("ht");

    fireEvent.change(urlInput, { target: { value: "http" } });
    expect(urlInput.value).toBe("http");

    fireEvent.change(urlInput, { target: { value: "https" } });
    expect(urlInput.value).toBe("https");

    fireEvent.change(urlInput, { target: { value: "https:" } });
    expect(urlInput.value).toBe("https:");

    fireEvent.change(urlInput, { target: { value: "https:/" } });
    expect(urlInput.value).toBe("https:/");

    fireEvent.change(urlInput, { target: { value: "https://" } });
    expect(urlInput.value).toBe("https://");

    // Continue typing the domain
    fireEvent.change(urlInput, { target: { value: "https://example.com" } });
    expect(urlInput.value).toBe("https://example.com");
  });

  it("should only show URL format error for incomplete URLs, not trailing slash error", async () => {
    const TestWrapper = () => {
      const form = useSSOSettingsForm("admin-panel");
      return (
        <SSOModals
          isAddSSOModalVisible={true}
          isInstructionsModalVisible={false}
          handleAddSSOOk={() => {}}
          handleAddSSOCancel={() => {}}
          handleShowInstructions={() => {}}
          handleInstructionsOk={() => {}}
          handleInstructionsCancel={() => {}}
          form={form}
          accessToken={null}
          ssoConfigured={false}
        />
      );
    };

    render(<TestWrapper />);

    // Find and interact with the SSO provider select
    await user().click(screen.getByLabelText("SSO Provider"));
    // Wait for dropdown and select Google
    const googleOption = await screen.findByText("Google SSO");
    await user().click(googleOption);

    // Fill in the email field
    const emailInput = screen.getByLabelText("Proxy Admin Email");
    fireEvent.change(emailInput, { target: { value: "test@example.com" } });

    // Fill in an incomplete URL like "http:"
    const urlInput = screen.getByLabelText("Proxy Base URL");
    fireEvent.change(urlInput, { target: { value: "http:" } });

    // Submit the form
    const saveButton = screen.getByText("Save");
    fireEvent.click(saveButton);

    // Check that only the URL format error appears (use findByText for async rendering)
    const errorMessage = await screen.findByText("URL must start with http:// or https://", {}, { timeout: 3000 });
    expect(errorMessage).toBeInTheDocument();

    // Verify the trailing slash error does NOT appear
    expect(screen.queryByText("URL must not end with a trailing slash")).not.toBeInTheDocument();
  });

  it("should load existing SSO settings when modal opens", async () => {
    const mockSSOData = {
      values: {
        google_client_id: "test-client-id",
        google_client_secret: "test-client-secret",
        proxy_base_url: "https://example.com",
        user_email: "admin@example.com",
        role_mappings: {
          group_claim: "groups",
          default_role: "internal_user",
          roles: {
            proxy_admin: ["admin-group"],
            proxy_admin_viewer: ["viewer-group"],
            internal_user: ["user-group"],
            internal_user_viewer: ["readonly-group"],
          },
        },
      },
    };

    (getSSOSettings as any).mockResolvedValue(mockSSOData);

    const TestWrapper = () => {
      const form = useSSOSettingsForm("admin-panel");

      return (
        <SSOModals
          isAddSSOModalVisible={true}
          isInstructionsModalVisible={false}
          handleAddSSOOk={() => {}}
          handleAddSSOCancel={() => {}}
          handleShowInstructions={() => {}}
          handleInstructionsOk={() => {}}
          handleInstructionsCancel={() => {}}
          form={form}
          accessToken="test-token"
          ssoConfigured={false}
        />
      );
    };

    render(<TestWrapper />);

    // Wait for the useEffect to load data and populate form
    await waitFor(() => {
      expect(getSSOSettings).toHaveBeenCalledWith("test-token");
    });

    // Check that form fields are populated with loaded data
    await waitFor(() => {
      const emailInput = screen.getByLabelText("Proxy Admin Email") as HTMLInputElement;
      expect(emailInput.value).toBe("admin@example.com");
    });

    const urlInput = screen.getByLabelText("Proxy Base URL") as HTMLInputElement;
    expect(urlInput.value).toBe("https://example.com");

    // Check that role mappings are populated
    const groupClaimInput = screen.getByLabelText("Group Claim") as HTMLInputElement;
    expect(groupClaimInput.value).toBe("groups");
  });

  it("should submit form with role mappings enabled", async () => {
    const mockHandleShowInstructions = vi.fn();
    (updateSSOSettings as any).mockResolvedValue({});
    // Mock getSSOSettings to return empty data so form starts clean
    (getSSOSettings as any).mockResolvedValue({ values: {} });

    let formInstance: ReturnType<typeof useSSOSettingsForm> | null = null;

    const TestWrapper = () => {
      const form = useSSOSettingsForm("admin-panel");
      formInstance = form;

      return (
        <SSOModals
          isAddSSOModalVisible={true}
          isInstructionsModalVisible={false}
          handleAddSSOOk={() => {}}
          handleAddSSOCancel={() => {}}
          handleShowInstructions={mockHandleShowInstructions}
          handleInstructionsOk={() => {}}
          handleInstructionsCancel={() => {}}
          form={form}
          accessToken="test-token"
          ssoConfigured={false}
        />
      );
    };

    render(<TestWrapper />);

    // Wait for any initial loading to complete
    await waitFor(() => {
      expect(getSSOSettings).toHaveBeenCalledWith("test-token");
    });

    // Set the provider directly using the form to trigger conditional rendering
    formInstance!.setValue("sso_provider", "okta");

    // Wait for the "Use Role Mappings" checkbox to appear
    await waitFor(() => {
      expect(screen.getAllByLabelText("Use Role Mappings")[0]).toBeInTheDocument();
    });

    // Enable role mappings
    const roleMappingsCheckbox = screen.getAllByLabelText("Use Role Mappings")[0];
    await user().click(roleMappingsCheckbox);

    // Fill required fields
    const emailInput = screen.getByLabelText("Proxy Admin Email");
    fireEvent.change(emailInput, { target: { value: "admin@example.com" } });

    const urlInput = screen.getByLabelText("Proxy Base URL");
    fireEvent.change(urlInput, { target: { value: "https://example.com" } });

    // Fill Okta specific fields
    const clientIdInput = screen.getByLabelText("Generic Client ID");
    fireEvent.change(clientIdInput, { target: { value: "test-client-id" } });

    const clientSecretInput = screen.getByLabelText("Generic Client Secret");
    fireEvent.change(clientSecretInput, { target: { value: "test-client-secret" } });

    const authEndpointInput = screen.getByLabelText("Authorization Endpoint");
    fireEvent.change(authEndpointInput, { target: { value: "https://example.okta.com/authorize" } });

    const tokenEndpointInput = screen.getByLabelText("Token Endpoint");
    fireEvent.change(tokenEndpointInput, { target: { value: "https://example.okta.com/token" } });

    const userinfoEndpointInput = screen.getByLabelText("Userinfo Endpoint");
    fireEvent.change(userinfoEndpointInput, { target: { value: "https://example.okta.com/userinfo" } });

    // Fill role mapping fields
    const groupClaimInput = screen.getByLabelText("Group Claim");
    fireEvent.change(groupClaimInput, { target: { value: "groups" } });

    const proxyAdminTeamsInput = screen.getByLabelText("Proxy Admin Teams");
    fireEvent.change(proxyAdminTeamsInput, { target: { value: "admin-group, super-admin" } });

    // Submit the form
    const saveButton = screen.getByText("Save");
    fireEvent.click(saveButton);

    // Verify the API was called with correct payload including role mappings
    await waitFor(() => {
      expect(updateSSOSettings).toHaveBeenCalledWith("test-token", {
        sso_provider: "okta",
        user_email: "admin@example.com",
        proxy_base_url: "https://example.com",
        generic_client_id: "test-client-id",
        generic_client_secret: "test-client-secret",
        generic_authorization_endpoint: "https://example.okta.com/authorize",
        generic_token_endpoint: "https://example.okta.com/token",
        generic_userinfo_endpoint: "https://example.okta.com/userinfo",
        role_mappings: {
          provider: "generic",
          group_claim: "groups",
          default_role: "internal_user",
          roles: {
            proxy_admin: ["admin-group", "super-admin"],
            proxy_admin_viewer: [],
            internal_user: [],
            internal_user_viewer: [],
          },
        },
      });
    });

    expect(mockHandleShowInstructions).toHaveBeenCalled();
  });

  it("should submit SAML settings with the unsolicited toggle mapped to a 'true'/'false' string", async () => {
    const mockHandleShowInstructions = vi.fn();
    vi.mocked(updateSSOSettings).mockResolvedValue({});
    vi.mocked(getSSOSettings).mockResolvedValue({ values: {} });

    let formInstance: ReturnType<typeof useSSOSettingsForm> | null = null;

    const TestWrapper = () => {
      const form = useSSOSettingsForm("admin-panel");
      formInstance = form;

      return (
        <SSOModals
          isAddSSOModalVisible={true}
          isInstructionsModalVisible={false}
          handleAddSSOOk={() => {}}
          handleAddSSOCancel={() => {}}
          handleShowInstructions={mockHandleShowInstructions}
          handleInstructionsOk={() => {}}
          handleInstructionsCancel={() => {}}
          form={form}
          accessToken="test-token"
          ssoConfigured={false}
        />
      );
    };

    render(<TestWrapper />);

    await waitFor(() => {
      expect(getSSOSettings).toHaveBeenCalledWith("test-token");
    });

    formInstance!.setValue("sso_provider", "saml");

    await waitFor(() => {
      expect(screen.getByLabelText("IdP Metadata URL")).toBeInTheDocument();
    });

    fireEvent.change(screen.getByLabelText("Proxy Admin Email"), {
      target: { value: "admin@example.com" },
    });
    fireEvent.change(screen.getByLabelText("Proxy Base URL"), {
      target: { value: "https://proxy.example.com" },
    });
    fireEvent.change(screen.getByLabelText("IdP Metadata URL"), {
      target: { value: "https://idp.example.com/metadata" },
    });
    fireEvent.change(screen.getByLabelText("SP Entity ID"), {
      target: { value: "https://proxy.example.com/sso/saml/metadata" },
    });
    await user().click(screen.getAllByLabelText("Allow IdP-initiated (unsolicited) responses")[0]);

    fireEvent.click(screen.getByText("Save"));

    await waitFor(() => {
      expect(updateSSOSettings).toHaveBeenCalledWith(
        "test-token",
        expect.objectContaining({
          sso_provider: "saml",
          saml_idp_metadata_url: "https://idp.example.com/metadata",
          saml_sp_entity_id: "https://proxy.example.com/sso/saml/metadata",
          saml_allow_unsolicited: "true",
        }),
      );
    });

    expect(mockHandleShowInstructions).toHaveBeenCalled();
  });

  it("should show Clear button and clear SSO settings when configured", async () => {
    const mockHandleAddSSOOk = vi.fn();
    (updateSSOSettings as any).mockResolvedValue({});
    (toast.success as any).mockImplementation(() => {});

    const TestWrapper = () => {
      const form = useSSOSettingsForm("admin-panel");

      return (
        <SSOModals
          isAddSSOModalVisible={true}
          isInstructionsModalVisible={false}
          handleAddSSOOk={mockHandleAddSSOOk}
          handleAddSSOCancel={() => {}}
          handleShowInstructions={() => {}}
          handleInstructionsOk={() => {}}
          handleInstructionsCancel={() => {}}
          form={form}
          accessToken="test-token"
          ssoConfigured={true}
        />
      );
    };

    render(<TestWrapper />);

    // Check that Clear button is visible when SSO is configured
    const clearButton = screen.getByText("Clear");
    expect(clearButton).toBeInTheDocument();

    // Click Clear button to open confirmation modal
    fireEvent.click(clearButton);

    // Confirm the clear action in the modal
    const confirmButton = screen.getByText("Yes, Clear");
    fireEvent.click(confirmButton);

    // Verify the clear API was called with null values
    await waitFor(() => {
      expect(updateSSOSettings).toHaveBeenCalledWith("test-token", {
        google_client_id: null,
        google_client_secret: null,
        microsoft_client_id: null,
        microsoft_client_secret: null,
        microsoft_tenant: null,
        generic_client_id: null,
        generic_client_secret: null,
        generic_authorization_endpoint: null,
        generic_token_endpoint: null,
        generic_userinfo_endpoint: null,
        saml_idp_metadata_url: null,
        saml_idp_metadata_xml: null,
        saml_sp_entity_id: null,
        saml_allow_unsolicited: null,
        generic_scope: null,
        proxy_base_url: null,
        user_email: null,
        sso_provider: null,
        role_mappings: null,
      });
    });

    expect(toast.success).toHaveBeenCalledWith("SSO settings cleared successfully");
    expect(mockHandleAddSSOOk).toHaveBeenCalled();
  });

  it("renders provider logos in the SSO provider dropdown", async () => {
    const TestWrapper = () => {
      const form = useSSOSettingsForm("admin-panel");
      return (
        <SSOModals
          isAddSSOModalVisible={true}
          isInstructionsModalVisible={false}
          handleAddSSOOk={() => {}}
          handleAddSSOCancel={() => {}}
          handleShowInstructions={() => {}}
          handleInstructionsOk={() => {}}
          handleInstructionsCancel={() => {}}
          form={form}
          accessToken={null}
          ssoConfigured={false}
        />
      );
    };

    render(<TestWrapper />);

    await user().click(screen.getByLabelText("SSO Provider"));

    await waitFor(() => {
      expect(screen.getAllByAltText("Google SSO logo").length).toBeGreaterThan(0);
    });

    expect(screen.getAllByAltText("Google SSO logo")[0]).toHaveAttribute("src", expect.stringContaining("google.svg"));
    expect(screen.getAllByAltText("Microsoft SSO logo")[0]).toHaveAttribute(
      "src",
      expect.stringContaining("microsoft_azure.svg"),
    );
    expect(screen.getAllByAltText("Okta / Auth0 SSO logo")[0]).toHaveAttribute(
      "src",
      expect.stringContaining("https://www.okta.com/"),
    );
    expect(screen.queryByAltText("Generic SSO logo")).not.toBeInTheDocument();
  });
});
