import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import SSOModals from "./SSOModals";

vi.mock("./networking", () => ({
  getSSOSettings: vi.fn(),
  updateSSOSettings: vi.fn(),
}));

vi.mock("./shared/errorUtils", () => ({
  parseErrorMessage: vi.fn((error) => error?.message || "An error occurred"),
}));

import NotificationsManager from "./molecules/notifications_manager";
import { getSSOSettings, updateSSOSettings } from "./networking";

interface ExternalFormHandle {
  resetFields: () => void;
  setFieldsValue: (values: Record<string, unknown>) => void;
  getFieldsValue: () => Record<string, unknown>;
}

function makeExternalForm(): ExternalFormHandle {
  return {
    resetFields: () => {},
    setFieldsValue: () => {},
    getFieldsValue: () => ({}),
  };
}

/** shadcn `Select` renders a hidden native `<select>` for form submission and
 * accessibility — use that to drive value changes reliably in jsdom. */
async function selectSSOProvider(trigger: HTMLElement, value: string) {
  // The hidden select is the next sibling of the Radix trigger inside the
  // wrapper. Locate it via the shared label association.
  const hiddenSelect = trigger.parentElement?.querySelector("select");
  if (!hiddenSelect) {
    throw new Error("Could not find hidden native select element");
  }
  await act(async () => {
    fireEvent.change(hiddenSelect, { target: { value } });
  });
}

describe("SSOModals", () => {
  it("should render the SSOModals component", () => {
    render(
      <SSOModals
        isAddSSOModalVisible={true}
        isInstructionsModalVisible={false}
        handleAddSSOOk={() => {}}
        handleAddSSOCancel={() => {}}
        handleShowInstructions={() => {}}
        handleInstructionsOk={() => {}}
        handleInstructionsCancel={() => {}}
        form={makeExternalForm()}
        accessToken={null}
        ssoConfigured={false}
      />,
    );
    expect(screen.getByText("Add SSO")).toBeInTheDocument();
  });

  it("should show validation error if proxy base url is not a valid URL", async () => {
    render(
      <SSOModals
        isAddSSOModalVisible={true}
        isInstructionsModalVisible={false}
        handleAddSSOOk={() => {}}
        handleAddSSOCancel={() => {}}
        handleShowInstructions={() => {}}
        handleInstructionsOk={() => {}}
        handleInstructionsCancel={() => {}}
        form={makeExternalForm()}
        accessToken={null}
        ssoConfigured={false}
      />,
    );

    const trigger = screen.getByLabelText("SSO Provider");
    await selectSSOProvider(trigger, "google");

    const emailInput = screen.getByLabelText(/Proxy Admin Email/);
    await act(async () => {
      fireEvent.change(emailInput, { target: { value: "test@example.com" } });
    });

    const urlInput = screen.getByLabelText(/Proxy Base URL/);
    await act(async () => {
      fireEvent.change(urlInput, { target: { value: "invalid-url" } });
    });

    const saveButton = screen.getByRole("button", { name: "Save" });
    await act(async () => {
      fireEvent.click(saveButton);
    });

    await waitFor(
      () => {
        expect(screen.getByText("URL must start with http:// or https://")).toBeInTheDocument();
      },
      { timeout: 5000 },
    );
  });

  it("should show validation error if proxy base url ends with trailing slash", async () => {
    render(
      <SSOModals
        isAddSSOModalVisible={true}
        isInstructionsModalVisible={false}
        handleAddSSOOk={() => {}}
        handleAddSSOCancel={() => {}}
        handleShowInstructions={() => {}}
        handleInstructionsOk={() => {}}
        handleInstructionsCancel={() => {}}
        form={makeExternalForm()}
        accessToken={null}
        ssoConfigured={false}
      />,
    );

    const trigger = screen.getByLabelText("SSO Provider");
    await selectSSOProvider(trigger, "google");

    const emailInput = screen.getByLabelText(/Proxy Admin Email/);
    await act(async () => {
      fireEvent.change(emailInput, { target: { value: "test@example.com" } });
    });

    const urlInput = screen.getByLabelText(/Proxy Base URL/) as HTMLInputElement;
    await act(async () => {
      fireEvent.change(urlInput, { target: { value: "https://example.com/" } });
    });

    const saveButton = screen.getByRole("button", { name: "Save" });
    await act(async () => {
      fireEvent.click(saveButton);
    });

    const errorMessage = await screen.findByText(
      "URL must not end with a trailing slash",
      {},
      { timeout: 5000 },
    );
    expect(errorMessage).toBeInTheDocument();
  });

  it("should allow typing https:// without interfering with slashes", async () => {
    render(
      <SSOModals
        isAddSSOModalVisible={true}
        isInstructionsModalVisible={false}
        handleAddSSOOk={() => {}}
        handleAddSSOCancel={() => {}}
        handleShowInstructions={() => {}}
        handleInstructionsOk={() => {}}
        handleInstructionsCancel={() => {}}
        form={makeExternalForm()}
        accessToken={null}
        ssoConfigured={false}
      />,
    );

    const urlInput = screen.getByLabelText(/Proxy Base URL/) as HTMLInputElement;
    const steps = ["h", "ht", "http", "https", "https:", "https:/", "https://", "https://example.com"];
    for (const v of steps) {
      await act(async () => {
        fireEvent.change(urlInput, { target: { value: v } });
      });
      expect(urlInput.value).toBe(v);
    }
  });

  it("should only show URL format error for incomplete URLs, not trailing slash error", async () => {
    render(
      <SSOModals
        isAddSSOModalVisible={true}
        isInstructionsModalVisible={false}
        handleAddSSOOk={() => {}}
        handleAddSSOCancel={() => {}}
        handleShowInstructions={() => {}}
        handleInstructionsOk={() => {}}
        handleInstructionsCancel={() => {}}
        form={makeExternalForm()}
        accessToken={null}
        ssoConfigured={false}
      />,
    );

    const trigger = screen.getByLabelText("SSO Provider");
    await selectSSOProvider(trigger, "google");

    const emailInput = screen.getByLabelText(/Proxy Admin Email/);
    await act(async () => {
      fireEvent.change(emailInput, { target: { value: "test@example.com" } });
    });

    const urlInput = screen.getByLabelText(/Proxy Base URL/);
    await act(async () => {
      fireEvent.change(urlInput, { target: { value: "http:" } });
    });

    const saveButton = screen.getByRole("button", { name: "Save" });
    await act(async () => {
      fireEvent.click(saveButton);
    });

    const errorMessage = await screen.findByText(
      "URL must start with http:// or https://",
      {},
      { timeout: 3000 },
    );
    expect(errorMessage).toBeInTheDocument();

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

    render(
      <SSOModals
        isAddSSOModalVisible={true}
        isInstructionsModalVisible={false}
        handleAddSSOOk={() => {}}
        handleAddSSOCancel={() => {}}
        handleShowInstructions={() => {}}
        handleInstructionsOk={() => {}}
        handleInstructionsCancel={() => {}}
        form={makeExternalForm()}
        accessToken="test-token"
        ssoConfigured={false}
      />,
    );

    await waitFor(() => {
      expect(getSSOSettings).toHaveBeenCalledWith("test-token");
    });

    await waitFor(() => {
      const emailInput = screen.getByLabelText(/Proxy Admin Email/) as HTMLInputElement;
      expect(emailInput.value).toBe("admin@example.com");
    });

    const urlInput = screen.getByLabelText(/Proxy Base URL/) as HTMLInputElement;
    expect(urlInput.value).toBe("https://example.com");

    const groupClaimInput = screen.getByLabelText(/Group Claim/) as HTMLInputElement;
    expect(groupClaimInput.value).toBe("groups");
  });

  it("should submit form with role mappings enabled", async () => {
    const mockHandleShowInstructions = vi.fn();
    (updateSSOSettings as any).mockResolvedValue({});
    (getSSOSettings as any).mockResolvedValue({ values: {} });

    render(
      <SSOModals
        isAddSSOModalVisible={true}
        isInstructionsModalVisible={false}
        handleAddSSOOk={() => {}}
        handleAddSSOCancel={() => {}}
        handleShowInstructions={mockHandleShowInstructions}
        handleInstructionsOk={() => {}}
        handleInstructionsCancel={() => {}}
        form={makeExternalForm()}
        accessToken="test-token"
        ssoConfigured={false}
      />,
    );

    await waitFor(() => {
      expect(getSSOSettings).toHaveBeenCalledWith("test-token");
    });

    const providerTrigger = screen.getByLabelText("SSO Provider");
    await selectSSOProvider(providerTrigger, "okta");

    await waitFor(() => {
      expect(screen.getByLabelText("Use Role Mappings")).toBeInTheDocument();
    });

    const roleMappingsCheckbox = screen.getByLabelText("Use Role Mappings");
    await act(async () => {
      fireEvent.click(roleMappingsCheckbox);
    });

    const emailInput = screen.getByLabelText(/Proxy Admin Email/);
    await act(async () => {
      fireEvent.change(emailInput, { target: { value: "admin@example.com" } });
    });

    const urlInput = screen.getByLabelText(/Proxy Base URL/);
    await act(async () => {
      fireEvent.change(urlInput, { target: { value: "https://example.com" } });
    });

    const clientIdInput = screen.getByLabelText(/Generic Client ID/);
    await act(async () => {
      fireEvent.change(clientIdInput, { target: { value: "test-client-id" } });
    });

    const clientSecretInput = screen.getByLabelText(/Generic Client Secret/);
    await act(async () => {
      fireEvent.change(clientSecretInput, { target: { value: "test-client-secret" } });
    });

    const authEndpointInput = screen.getByLabelText(/Authorization Endpoint/);
    await act(async () => {
      fireEvent.change(authEndpointInput, {
        target: { value: "https://example.okta.com/authorize" },
      });
    });

    const tokenEndpointInput = screen.getByLabelText(/Token Endpoint/);
    await act(async () => {
      fireEvent.change(tokenEndpointInput, {
        target: { value: "https://example.okta.com/token" },
      });
    });

    const userinfoEndpointInput = screen.getByLabelText(/Userinfo Endpoint/);
    await act(async () => {
      fireEvent.change(userinfoEndpointInput, {
        target: { value: "https://example.okta.com/userinfo" },
      });
    });

    const groupClaimInput = screen.getByLabelText(/Group Claim/);
    await act(async () => {
      fireEvent.change(groupClaimInput, { target: { value: "groups" } });
    });

    const proxyAdminTeamsInput = screen.getByLabelText(/Proxy Admin Teams/);
    await act(async () => {
      fireEvent.change(proxyAdminTeamsInput, {
        target: { value: "admin-group, super-admin" },
      });
    });

    const saveButton = screen.getByRole("button", { name: "Save" });
    await act(async () => {
      fireEvent.click(saveButton);
    });

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
    (NotificationsManager.success as any).mockImplementation?.(() => {});

    render(
      <SSOModals
        isAddSSOModalVisible={true}
        isInstructionsModalVisible={false}
        handleAddSSOOk={mockHandleAddSSOOk}
        handleAddSSOCancel={() => {}}
        handleShowInstructions={() => {}}
        handleInstructionsOk={() => {}}
        handleInstructionsCancel={() => {}}
        form={makeExternalForm()}
        accessToken="test-token"
        ssoConfigured={true}
      />,
    );

    const clearButton = screen.getByRole("button", { name: "Clear" });
    expect(clearButton).toBeInTheDocument();

    await act(async () => {
      fireEvent.click(clearButton);
    });

    const confirmButton = await screen.findByRole("button", { name: "Yes, Clear" });
    await act(async () => {
      fireEvent.click(confirmButton);
    });

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
