import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, Mock } from "vitest";
import EditSSOSettingsModal from "./EditSSOSettingsModal";
import { useSSOSettings } from "@/app/(dashboard)/hooks/sso/useSSOSettings";
import { useEditSSOSettings } from "@/app/(dashboard)/hooks/sso/useEditSSOSettings";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { processSSOSettingsPayload } from "../utils";

// Constants
const SSO_PROVIDERS = {
  GOOGLE: "google",
  MICROSOFT: "microsoft",
  OKTA: "okta",
  AUTH0: "auth0",
  GENERIC: "generic",
} as const;

const TEST_DATA = {
  MODAL_TITLE: "Edit SSO Settings",
  MODAL_WIDTH: "800",
  SUCCESS_MESSAGE: "SSO settings updated successfully",
  ERROR_MESSAGE_PREFIX: "Failed to save SSO settings:",
  BUTTON_TEXT: {
    CANCEL: "Cancel",
    SAVE: "Save",
    SAVING: "Saving...",
  },
} as const;

const TEST_IDS = {
  MODAL: "modal",
  BUTTON: "button",
  BASE_SSO_FORM: "base-sso-form",
  TRIGGER_FORM_SUBMIT: "trigger-form-submit",
} as const;

// Mock form instance
const mockForm = {
  resetFields: vi.fn(),
  setFieldsValue: vi.fn(),
  getFieldsValue: vi.fn(),
  submit: vi.fn(),
};

// Types
type SSOData = {
  values: Record<string, any>;
} & Record<string, any>;

type SSOSettingsHookReturn = {
  data: SSOData | null;
  isLoading: boolean;
  error: any;
};

type EditSSOSettingsHookReturn = {
  mutateAsync: ReturnType<typeof vi.fn>;
  isPending: boolean;
};

// Test data factories
const createSSOData = (overrides: Record<string, any> = {}): SSOData => ({
  values: {
    user_email: "test@example.com",
    ...overrides,
  },
});

const createGoogleSSOData = (overrides: Record<string, any> = {}) =>
  createSSOData({
    google_client_id: "test-google-id",
    google_client_secret: "test-google-secret",
    ...overrides,
  });

const createMicrosoftSSOData = (overrides: Record<string, any> = {}) =>
  createSSOData({
    microsoft_client_id: "test-microsoft-id",
    microsoft_client_secret: "test-microsoft-secret",
    microsoft_tenant: "test-tenant",
    ...overrides,
  });

const createGenericSSOData = (overrides: Record<string, any> = {}) =>
  createSSOData({
    generic_client_id: "test-generic-id",
    generic_client_secret: "test-generic-secret",
    generic_authorization_endpoint: overrides.authorization_endpoint || "https://custom.example.com/oauth",
    ...overrides,
  });

const createRoleMappingsSSOData = (overrides: Record<string, any> = {}) =>
  createGoogleSSOData({
    role_mappings: {
      group_claim: "groups",
      default_role: "internal_user",
      roles: {
        proxy_admin: overrides.proxy_admin || ["admin-group"],
        proxy_admin_viewer: overrides.proxy_admin_viewer || ["viewer-group"],
        internal_user: overrides.internal_user || ["user-group"],
        internal_user_viewer: overrides.internal_user_viewer || ["readonly-group"],
      },
    },
    ...overrides,
  });

const createTeamMappingsSSOData = (overrides: Record<string, any> = {}) =>
  createGenericSSOData({
    team_mappings: {
      team_ids_jwt_field: overrides.team_ids_jwt_field || "teams",
    },
    ...overrides,
  });

// Mock utilities
const createMockHooks = (): {
  useSSOSettings: SSOSettingsHookReturn;
  useEditSSOSettings: EditSSOSettingsHookReturn;
} => ({
  useSSOSettings: {
    data: null,
    isLoading: false,
    error: null,
  },
  useEditSSOSettings: {
    mutateAsync: vi.fn(),
    isPending: false,
  },
});

vi.mock("antd", () => ({
  Modal: ({ children, open, title, footer, onCancel, width, ...props }: any) => (
    <div data-testid={TEST_IDS.MODAL} data-open={open} data-title={title} data-width={width} {...props}>
      <div data-testid="modal-content">{children}</div>
      <div data-testid="modal-footer">{footer}</div>
      <button data-testid="modal-cancel" onClick={onCancel} />
    </div>
  ),
  Button: ({ children, onClick, loading, disabled, ...props }: any) => (
    <button data-testid={TEST_IDS.BUTTON} onClick={onClick} data-loading={loading} disabled={disabled} {...props}>
      {children}
    </button>
  ),
  Form: {
    useForm: () => [mockForm],
  },
  Space: ({ children, ...props }: any) => (
    <div data-testid="space" {...props}>
      {children}
    </div>
  ),
}));

vi.mock("./BaseSSOSettingsForm", () => ({
  default: ({ form, onFormSubmit }: any) => (
    <div data-testid={TEST_IDS.BASE_SSO_FORM}>
      <button data-testid={TEST_IDS.TRIGGER_FORM_SUBMIT} onClick={() => onFormSubmit({ testField: "testValue" })}>
        Trigger Form Submit
      </button>
    </div>
  ),
}));

vi.mock("@/app/(dashboard)/hooks/sso/useSSOSettings", () => ({
  useSSOSettings: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/sso/useEditSSOSettings", () => ({
  useEditSSOSettings: vi.fn(),
}));

vi.mock("@/components/molecules/notifications_manager", () => ({
  default: {
    success: vi.fn(),
    fromBackend: vi.fn(),
  },
}));

vi.mock("@/components/shared/errorUtils", () => ({
  parseErrorMessage: vi.fn(),
}));

vi.mock("../utils", () => ({
  processSSOSettingsPayload: vi.fn(),
}));

// Test helpers
const setupMocks = (
  overrides: Partial<{
    useSSOSettings: Partial<SSOSettingsHookReturn>;
    useEditSSOSettings: Partial<EditSSOSettingsHookReturn>;
  }> = {},
) => {
  const defaultMocks = createMockHooks();
  const mocks = {
    useSSOSettings: { ...defaultMocks.useSSOSettings, ...overrides.useSSOSettings },
    useEditSSOSettings: { ...defaultMocks.useEditSSOSettings, ...overrides.useEditSSOSettings },
  };

  (useSSOSettings as Mock).mockReturnValue(mocks.useSSOSettings);
  (useEditSSOSettings as Mock).mockReturnValue(mocks.useEditSSOSettings);

  return mocks;
};

const renderComponent = (props: Partial<React.ComponentProps<typeof EditSSOSettingsModal>> = {}) => {
  const defaultProps = {
    isVisible: true,
    onCancel: vi.fn(),
    onSuccess: vi.fn(),
  };

  return {
    ...render(<EditSSOSettingsModal {...defaultProps} {...props} />),
    mockOnCancel: defaultProps.onCancel,
    mockOnSuccess: defaultProps.onSuccess,
  };
};

const getButtons = () => screen.getAllByTestId(TEST_IDS.BUTTON);
const getCancelButton = () => getButtons()[0];
const getSaveButton = () => getButtons()[1];

describe("EditSSOSettingsModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupMocks();
  });

  describe("Rendering", () => {
    it("renders without crashing", () => {
      expect(() => renderComponent()).not.toThrow();
    });

    it("displays modal with correct configuration", () => {
      renderComponent();

      const modal = screen.getByTestId(TEST_IDS.MODAL);
      expect(modal).toHaveAttribute("data-open", "true");
      expect(modal).toHaveAttribute("data-title", TEST_DATA.MODAL_TITLE);
      expect(modal).toHaveAttribute("data-width", TEST_DATA.MODAL_WIDTH);
    });

    it("displays modal as closed when not visible", () => {
      renderComponent({ isVisible: false });

      const modal = screen.getByTestId(TEST_IDS.MODAL);
      expect(modal).toHaveAttribute("data-open", "false");
    });
  });

  describe("Footer Actions", () => {
    it("renders cancel and save buttons", () => {
      renderComponent();

      const buttons = getButtons();
      expect(buttons).toHaveLength(2);
      expect(buttons[0]).toHaveTextContent(TEST_DATA.BUTTON_TEXT.CANCEL);
      expect(buttons[1]).toHaveTextContent(TEST_DATA.BUTTON_TEXT.SAVE);
    });

    it("calls onCancel and resets form when cancel button is clicked", () => {
      const { mockOnCancel } = renderComponent();

      fireEvent.click(getCancelButton());

      expect(mockForm.resetFields).toHaveBeenCalled();
      expect(mockOnCancel).toHaveBeenCalled();
    });

    it("calls form.submit when save button is clicked", () => {
      renderComponent();

      fireEvent.click(getSaveButton());

      expect(mockForm.submit).toHaveBeenCalled();
    });

    describe("Loading States", () => {
      it("disables cancel button during submission", () => {
        setupMocks({
          useEditSSOSettings: { mutateAsync: vi.fn(), isPending: true },
        });

        renderComponent();

        expect(getCancelButton()).toBeDisabled();
      });

      it("shows loading state on save button during submission", () => {
        setupMocks({
          useEditSSOSettings: { mutateAsync: vi.fn(), isPending: true },
        });

        renderComponent();

        expect(getSaveButton()).toHaveAttribute("data-loading", "true");
        expect(getSaveButton()).toHaveTextContent(TEST_DATA.BUTTON_TEXT.SAVING);
      });
    });
  });

  describe("Form Submission", () => {
    const formValues = { testField: "testValue" };
    const processedPayload = { processed: "payload" };

    beforeEach(() => {
      (processSSOSettingsPayload as any).mockReturnValue(processedPayload);
    });

    it("processes form values and submits successfully", async () => {
      const mockMutateAsync = vi.fn().mockImplementation((payload, options) => {
        options.onSuccess();
        return Promise.resolve({ success: true });
      });

      setupMocks({
        useEditSSOSettings: { mutateAsync: mockMutateAsync, isPending: false },
      });

      const { mockOnSuccess } = renderComponent();

      fireEvent.click(screen.getByTestId(TEST_IDS.TRIGGER_FORM_SUBMIT));

      expect(processSSOSettingsPayload).toHaveBeenCalledWith(formValues);
      expect(mockMutateAsync).toHaveBeenCalledWith(
        processedPayload,
        expect.objectContaining({
          onSuccess: expect.any(Function),
          onError: expect.any(Function),
        }),
      );
    });

    it("shows success notification and calls onSuccess callback", async () => {
      const mockMutateAsync = vi.fn().mockImplementation((payload, options) => {
        options.onSuccess();
        return Promise.resolve({ success: true });
      });

      setupMocks({
        useEditSSOSettings: { mutateAsync: mockMutateAsync, isPending: false },
      });

      const { mockOnSuccess } = renderComponent();

      fireEvent.click(screen.getByTestId(TEST_IDS.TRIGGER_FORM_SUBMIT));

      expect(NotificationsManager.success).toHaveBeenCalledWith(TEST_DATA.SUCCESS_MESSAGE);
      expect(mockOnSuccess).toHaveBeenCalled();
    });

    it("handles submission errors gracefully", async () => {
      const error = new Error("Submission failed");
      const mockMutateAsync = vi.fn().mockImplementation((payload, options) => {
        options.onError(error);
        return Promise.reject(error);
      });

      setupMocks({
        useEditSSOSettings: { mutateAsync: mockMutateAsync, isPending: false },
      });

      (parseErrorMessage as any).mockReturnValue("Parsed error message");

      renderComponent();

      fireEvent.click(screen.getByTestId(TEST_IDS.TRIGGER_FORM_SUBMIT));

      expect(parseErrorMessage).toHaveBeenCalledWith(error);
      expect(NotificationsManager.fromBackend).toHaveBeenCalledWith(
        `${TEST_DATA.ERROR_MESSAGE_PREFIX} Parsed error message`,
      );
    });
  });

  describe("Form Initialization", () => {
    describe("Provider Detection", () => {
      const testProviderDetection = (testName: string, ssoData: SSOData, expectedProvider: string) => {
        it(`detects ${testName} provider`, async () => {
          setupMocks({
            useSSOSettings: { data: ssoData, isLoading: false, error: null },
          });

          renderComponent();

          await waitFor(() => {
            expect(mockForm.setFieldsValue).toHaveBeenCalledWith({
              sso_provider: expectedProvider,
              ...ssoData.values,
            });
          });
        });
      };

      testProviderDetection("Google", createGoogleSSOData(), SSO_PROVIDERS.GOOGLE);

      testProviderDetection("Microsoft", createMicrosoftSSOData(), SSO_PROVIDERS.MICROSOFT);

      testProviderDetection(
        "Okta",
        createGenericSSOData({
          authorization_endpoint: "https://okta.example.com/oauth2/authorize",
        }),
        SSO_PROVIDERS.OKTA,
      );

      testProviderDetection(
        "Auth0 (detected as Okta)",
        createGenericSSOData({
          authorization_endpoint: "https://auth0.example.com/authorize",
        }),
        SSO_PROVIDERS.OKTA, // Auth0 URLs are detected as Okta provider
      );

      testProviderDetection("generic", createGenericSSOData(), SSO_PROVIDERS.GENERIC);
    });

    describe("Role Mappings", () => {
      it("processes role mappings with all roles assigned", async () => {
        const ssoData = createRoleMappingsSSOData();

        setupMocks({
          useSSOSettings: { data: ssoData, isLoading: false, error: null },
        });

        renderComponent();

        await waitFor(() => {
          expect(mockForm.setFieldsValue).toHaveBeenCalledWith({
            sso_provider: SSO_PROVIDERS.GOOGLE,
            ...ssoData.values,
            use_role_mappings: true,
            group_claim: "groups",
            default_role: "internal_user",
            proxy_admin_teams: "admin-group",
            admin_viewer_teams: "viewer-group",
            internal_user_teams: "user-group",
            internal_viewer_teams: "readonly-group",
          });
        });
      });

      it("handles empty role mapping arrays", async () => {
        const ssoData = createRoleMappingsSSOData({
          proxy_admin: [],
          proxy_admin_viewer: [],
          internal_user_viewer: [],
        });

        setupMocks({
          useSSOSettings: { data: ssoData, isLoading: false, error: null },
        });

        renderComponent();

        await waitFor(() => {
          expect(mockForm.setFieldsValue).toHaveBeenCalledWith({
            sso_provider: SSO_PROVIDERS.GOOGLE,
            ...ssoData.values,
            use_role_mappings: true,
            group_claim: "groups",
            default_role: "internal_user",
            proxy_admin_teams: "",
            admin_viewer_teams: "",
            internal_user_teams: "user-group",
            internal_viewer_teams: "",
          });
        });
      });
    });

    describe("Initialization Guards", () => {
      it("resets form before setting values", async () => {
        const ssoData = createGoogleSSOData();

        setupMocks({
          useSSOSettings: { data: ssoData, isLoading: false, error: null },
        });

        renderComponent();

        await waitFor(() => {
          expect(mockForm.resetFields).toHaveBeenCalled();
          expect(mockForm.setFieldsValue).toHaveBeenCalled();
        });
      });

      it("skips initialization when modal is not visible", () => {
        const ssoData = createGoogleSSOData();

        setupMocks({
          useSSOSettings: { data: ssoData, isLoading: false, error: null },
        });

        renderComponent({ isVisible: false });

        expect(mockForm.setFieldsValue).not.toHaveBeenCalled();
      });

      it("skips initialization when SSO data is unavailable", () => {
        setupMocks({
          useSSOSettings: { data: null, isLoading: false, error: null },
        });

        renderComponent();

        expect(mockForm.setFieldsValue).not.toHaveBeenCalled();
      });
    });
  });

  describe("Error Handling", () => {
    it("handles form submission errors with undefined error message", async () => {
      const error = new Error("Network error");
      const mockMutateAsync = vi.fn().mockImplementation((payload, options) => {
        options.onError(error);
        return Promise.reject(error);
      });

      setupMocks({
        useEditSSOSettings: { mutateAsync: mockMutateAsync, isPending: false },
      });

      (parseErrorMessage as any).mockReturnValue(undefined);

      renderComponent();

      fireEvent.click(screen.getByTestId(TEST_IDS.TRIGGER_FORM_SUBMIT));

      expect(NotificationsManager.fromBackend).toHaveBeenCalledWith(`${TEST_DATA.ERROR_MESSAGE_PREFIX} undefined`);
    });

    it("handles form submission with malformed data", async () => {
      const mockMutateAsync = vi.fn().mockImplementation((payload, options) => {
        options.onError(new Error("Invalid data"));
        return Promise.reject(new Error("Invalid data"));
      });

      setupMocks({
        useEditSSOSettings: { mutateAsync: mockMutateAsync, isPending: false },
      });

      (processSSOSettingsPayload as any).mockImplementation(() => {
        throw new Error("Processing failed");
      });

      renderComponent();

      fireEvent.click(screen.getByTestId(TEST_IDS.TRIGGER_FORM_SUBMIT));

      expect(processSSOSettingsPayload).toHaveBeenCalled();
      expect(mockMutateAsync).not.toHaveBeenCalled();
    });
  });

  describe("Edge Cases", () => {
    it("handles role mappings with undefined roles object", async () => {
      const ssoData = createGoogleSSOData({
        role_mappings: {
          group_claim: "groups",
          default_role: "internal_user",
          // roles is undefined
        },
      });

      setupMocks({
        useSSOSettings: { data: ssoData, isLoading: false, error: null },
      });

      renderComponent();

      await waitFor(() => {
        expect(mockForm.setFieldsValue).toHaveBeenCalledWith({
          sso_provider: SSO_PROVIDERS.GOOGLE,
          ...ssoData.values,
          use_role_mappings: true,
          group_claim: "groups",
          default_role: "internal_user",
          proxy_admin_teams: "",
          admin_viewer_teams: "",
          internal_user_teams: "",
          internal_viewer_teams: "",
        });
      });
    });
  });

  describe("Team Mappings", () => {
    it("processes team mappings when team_mappings exists", async () => {
      const ssoData = createTeamMappingsSSOData();

      setupMocks({
        useSSOSettings: { data: ssoData, isLoading: false, error: null },
      });

      renderComponent();

      await waitFor(() => {
        expect(mockForm.setFieldsValue).toHaveBeenCalledWith({
          sso_provider: SSO_PROVIDERS.GENERIC,
          ...ssoData.values,
          use_team_mappings: true,
          team_ids_jwt_field: "teams",
        });
      });
    });

    it("handles team mappings with custom JWT field name", async () => {
      const ssoData = createTeamMappingsSSOData({
        team_ids_jwt_field: "custom_teams_field",
      });

      setupMocks({
        useSSOSettings: { data: ssoData, isLoading: false, error: null },
      });

      renderComponent();

      await waitFor(() => {
        expect(mockForm.setFieldsValue).toHaveBeenCalledWith({
          sso_provider: SSO_PROVIDERS.GENERIC,
          ...ssoData.values,
          use_team_mappings: true,
          team_ids_jwt_field: "custom_teams_field",
        });
      });
    });

    it("handles team mappings and role mappings together", async () => {
      const ssoData = createGenericSSOData({
        role_mappings: {
          group_claim: "groups",
          default_role: "internal_user",
          roles: {
            proxy_admin: ["admin-group"],
            proxy_admin_viewer: [],
            internal_user: [],
            internal_user_viewer: [],
          },
        },
        team_mappings: {
          team_ids_jwt_field: "teams",
        },
      });

      setupMocks({
        useSSOSettings: { data: ssoData, isLoading: false, error: null },
      });

      renderComponent();

      await waitFor(() => {
        expect(mockForm.setFieldsValue).toHaveBeenCalledWith({
          sso_provider: SSO_PROVIDERS.GENERIC,
          ...ssoData.values,
          use_role_mappings: true,
          group_claim: "groups",
          default_role: "internal_user",
          proxy_admin_teams: "admin-group",
          admin_viewer_teams: "",
          internal_user_teams: "",
          internal_viewer_teams: "",
          use_team_mappings: true,
          team_ids_jwt_field: "teams",
        });
      });
    });

    it("does not set team mapping fields when team_mappings is not present", async () => {
      const ssoData = createGenericSSOData();

      setupMocks({
        useSSOSettings: { data: ssoData, isLoading: false, error: null },
      });

      renderComponent();

      await waitFor(() => {
        const callArgs = mockForm.setFieldsValue.mock.calls[0][0];
        expect(callArgs.use_team_mappings).toBeUndefined();
        expect(callArgs.team_ids_jwt_field).toBeUndefined();
      });
    });

    it("handles provider detection with partial SSO data", async () => {
      const ssoData = createSSOData({
        // Only has generic fields, no specific provider identifiers
        generic_client_id: "test-id",
        generic_authorization_endpoint: "https://unknown.provider.com/auth",
      });

      setupMocks({
        useSSOSettings: { data: ssoData, isLoading: false, error: null },
      });

      renderComponent();

      await waitFor(() => {
        expect(mockForm.setFieldsValue).toHaveBeenCalledWith({
          sso_provider: SSO_PROVIDERS.GENERIC,
          ...ssoData.values,
        });
      });
    });

    it("handles form submission when processing throws error", async () => {
      setupMocks({
        useEditSSOSettings: { mutateAsync: vi.fn(), isPending: false },
      });

      (processSSOSettingsPayload as any).mockImplementation(() => {
        throw new Error("Processing error");
      });

      renderComponent();

      expect(() => {
        fireEvent.click(screen.getByTestId(TEST_IDS.TRIGGER_FORM_SUBMIT));
      }).not.toThrow();

      expect(processSSOSettingsPayload).toHaveBeenCalled();
    });
  });
});
