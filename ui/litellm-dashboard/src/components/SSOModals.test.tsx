import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { Form } from "antd";
import { describe, expect, it, vi } from "vitest";
import SSOModals from "./SSOModals";

// Mock the networking functions
vi.mock("./networking", () => ({
  getSSOSettings: vi.fn(),
  updateSSOSettings: vi.fn(),
}));

// Mock parseErrorMessage
vi.mock("./shared/errorUtils", () => ({
  parseErrorMessage: vi.fn((error) => error?.message || "An error occurred"),
}));

import NotificationsManager from "./molecules/notifications_manager";
import { getSSOSettings, updateSSOSettings } from "./networking";

describe("SSOModals", () => {
  it("should render the SSOModals component", () => {
    const TestWrapper = () => {
      const [form] = Form.useForm();

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
      const [form] = Form.useForm();
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
    const ssoProviderSelect = screen.getByLabelText("SSO Provider");
    fireEvent.mouseDown(ssoProviderSelect);
    // Wait for dropdown and select Google
    await waitFor(() => {
      const googleOption = screen.getByText("Google SSO");
      fireEvent.click(googleOption);
    });

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
      const [form] = Form.useForm();
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
    const ssoProviderSelect = screen.getByLabelText("SSO Provider");
    fireEvent.mouseDown(ssoProviderSelect);
    // Wait for dropdown and select Google
    await waitFor(() => {
      const googleOption = screen.getByText("Google SSO");
      fireEvent.click(googleOption);
    });

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
      const [form] = Form.useForm();
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
      const [form] = Form.useForm();
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
    const ssoProviderSelect = screen.getByLabelText("SSO Provider");
    fireEvent.mouseDown(ssoProviderSelect);
    // Wait for dropdown and select Google
    await waitFor(() => {
      const googleOption = screen.getByText("Google SSO");
      fireEvent.click(googleOption);
    });

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
      const [form] = Form.useForm();

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

    let formInstance: any = null;

    const TestWrapper = () => {
      const [form] = Form.useForm();
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
    formInstance.setFieldsValue({ sso_provider: "okta" });

    // Wait for the "Use Role Mappings" checkbox to appear
    await waitFor(() => {
      expect(screen.getByLabelText("Use Role Mappings")).toBeInTheDocument();
    });

    // Enable role mappings
    const roleMappingsCheckbox = screen.getByLabelText("Use Role Mappings");
    fireEvent.click(roleMappingsCheckbox);

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

  it("should show Clear button and clear SSO settings when configured", async () => {
    const mockHandleAddSSOOk = vi.fn();
    (updateSSOSettings as any).mockResolvedValue({});
    (NotificationsManager.success as any).mockImplementation(() => {});

    const TestWrapper = () => {
      const [form] = Form.useForm();

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
        proxy_base_url: null,
        user_email: null,
        sso_provider: null,
        role_mappings: null,
      });
    });

    expect(NotificationsManager.success).toHaveBeenCalledWith("SSO settings cleared successfully");
    expect(mockHandleAddSSOOk).toHaveBeenCalled();
  });
});
