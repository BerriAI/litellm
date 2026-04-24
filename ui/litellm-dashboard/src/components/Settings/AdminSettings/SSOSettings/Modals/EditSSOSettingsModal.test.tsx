import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { useFormContext } from "react-hook-form";
import { beforeEach, describe, expect, it, Mock, vi } from "vitest";

import EditSSOSettingsModal from "./EditSSOSettingsModal";
import { useSSOSettings } from "@/app/(dashboard)/hooks/sso/useSSOSettings";
import { useEditSSOSettings } from "@/app/(dashboard)/hooks/sso/useEditSSOSettings";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { processSSOSettingsPayload } from "../utils";

const SSO_PROVIDERS = {
  GOOGLE: "google",
  MICROSOFT: "microsoft",
  OKTA: "okta",
  AUTH0: "auth0",
  GENERIC: "generic",
} as const;

const TEST_DATA = {
  MODAL_TITLE: "Edit SSO Settings",
  SUCCESS_MESSAGE: "SSO settings updated successfully",
  ERROR_MESSAGE_PREFIX: "Failed to save SSO settings:",
  BUTTON_TEXT: {
    CANCEL: "Cancel",
    SAVE: "Save",
    SAVING: "Saving...",
  },
} as const;

const TRIGGER_SUBMIT_TEST_ID = "trigger-form-submit";
const FORM_VALUES_TEST_ID = "captured-form-values";

type SSOData = {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  values: Record<string, any>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
} & Record<string, any>;

type SSOSettingsHookReturn = {
  data: SSOData | null;
  isLoading: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  error: any;
};

type EditSSOSettingsHookReturn = {
  mutateAsync: ReturnType<typeof vi.fn>;
  isPending: boolean;
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const createSSOData = (overrides: Record<string, any> = {}): SSOData => ({
  values: {
    user_email: "test@example.com",
    ...overrides,
  },
});

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const createGoogleSSOData = (overrides: Record<string, any> = {}) =>
  createSSOData({
    google_client_id: "test-google-id",
    google_client_secret: "test-google-secret",
    ...overrides,
  });

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const createMicrosoftSSOData = (overrides: Record<string, any> = {}) =>
  createSSOData({
    microsoft_client_id: "test-microsoft-id",
    microsoft_client_secret: "test-microsoft-secret",
    microsoft_tenant: "test-tenant",
    ...overrides,
  });

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const createGenericSSOData = (overrides: Record<string, any> = {}) =>
  createSSOData({
    generic_client_id: "test-generic-id",
    generic_client_secret: "test-generic-secret",
    generic_authorization_endpoint:
      overrides.authorization_endpoint || "https://custom.example.com/oauth",
    ...overrides,
  });

// eslint-disable-next-line @typescript-eslint/no-explicit-any
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

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const createTeamMappingsSSOData = (overrides: Record<string, any> = {}) =>
  createGenericSSOData({
    team_mappings: {
      team_ids_jwt_field: overrides.team_ids_jwt_field || "teams",
    },
    ...overrides,
  });

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

// Replaces the real BaseSSOSettingsForm with a spy that:
//   1. exposes all RHF form values via a hidden JSON blob for assertion
//   2. provides a test-id button that submits the parent form so we can
//      exercise the submit path without rendering the full provider UI.
vi.mock("./BaseSSOSettingsForm", () => {
  const Mock = () => {
    const ctx = useFormContext();
    const values = ctx.watch();
    return (
      <div data-testid="base-sso-form">
        <span data-testid={FORM_VALUES_TEST_ID}>{JSON.stringify(values)}</span>
      </div>
    );
  };
  return { default: Mock };
});

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

const setupMocks = (
  overrides: Partial<{
    useSSOSettings: Partial<SSOSettingsHookReturn>;
    useEditSSOSettings: Partial<EditSSOSettingsHookReturn>;
  }> = {},
) => {
  const defaultMocks = createMockHooks();
  const mocks = {
    useSSOSettings: {
      ...defaultMocks.useSSOSettings,
      ...overrides.useSSOSettings,
    },
    useEditSSOSettings: {
      ...defaultMocks.useEditSSOSettings,
      ...overrides.useEditSSOSettings,
    },
  };

  (useSSOSettings as unknown as Mock).mockReturnValue(mocks.useSSOSettings);
  (useEditSSOSettings as unknown as Mock).mockReturnValue(mocks.useEditSSOSettings);

  return mocks;
};

const renderComponent = (
  props: Partial<React.ComponentProps<typeof EditSSOSettingsModal>> = {},
) => {
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

const getCancelButton = () =>
  screen.getByRole("button", { name: TEST_DATA.BUTTON_TEXT.CANCEL });
const getSaveButton = () =>
  screen.getByRole("button", {
    name: new RegExp(
      `^(${TEST_DATA.BUTTON_TEXT.SAVE}|${TEST_DATA.BUTTON_TEXT.SAVING})$`,
    ),
  });

// Reads the JSON blob that the mocked BaseSSOSettingsForm renders.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const getCapturedFormValues = (): Record<string, any> => {
  const el = screen.queryByTestId(FORM_VALUES_TEST_ID);
  if (!el) return {};
  return JSON.parse(el.textContent || "{}");
};

describe("EditSSOSettingsModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setupMocks();
  });

  describe("Rendering", () => {
    it("should render without crashing", () => {
      expect(() => renderComponent()).not.toThrow();
    });

    it("should display dialog with correct title when visible", () => {
      renderComponent();

      expect(screen.getByRole("dialog")).toBeInTheDocument();
      expect(screen.getByText(TEST_DATA.MODAL_TITLE)).toBeInTheDocument();
    });

    it("should not render dialog when not visible", () => {
      renderComponent({ isVisible: false });

      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });
  });

  describe("Footer Actions", () => {
    it("should render cancel and save buttons", () => {
      renderComponent();

      expect(getCancelButton()).toBeInTheDocument();
      expect(getSaveButton()).toBeInTheDocument();
    });

    it("should call onCancel when cancel button is clicked", () => {
      const { mockOnCancel } = renderComponent();

      fireEvent.click(getCancelButton());

      expect(mockOnCancel).toHaveBeenCalled();
    });

    describe("Loading States", () => {
      it("should disable cancel button during submission", () => {
        setupMocks({
          useEditSSOSettings: { mutateAsync: vi.fn(), isPending: true },
        });

        renderComponent();

        expect(getCancelButton()).toBeDisabled();
      });

      it("should show Saving text on save button during submission", () => {
        setupMocks({
          useEditSSOSettings: { mutateAsync: vi.fn(), isPending: true },
        });

        renderComponent();

        expect(
          screen.getByRole("button", { name: TEST_DATA.BUTTON_TEXT.SAVING }),
        ).toBeInTheDocument();
      });
    });
  });

  describe("Form Submission", () => {
    const processedPayload = { processed: "payload" };

    beforeEach(() => {
      (processSSOSettingsPayload as unknown as Mock).mockReturnValue(
        processedPayload,
      );
    });

    it("should process form values and submits successfully", async () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const mockMutateAsync = vi.fn().mockImplementation((payload: any, options: any) => {
        options.onSuccess();
        return Promise.resolve({ success: true });
      });

      setupMocks({
        useSSOSettings: {
          data: createGoogleSSOData(),
          isLoading: false,
          error: null,
        },
        useEditSSOSettings: {
          mutateAsync: mockMutateAsync,
          isPending: false,
        },
      });

      renderComponent();

      await waitFor(() => {
        expect(getCapturedFormValues().sso_provider).toBe(SSO_PROVIDERS.GOOGLE);
      });

      fireEvent.click(getSaveButton());

      await waitFor(() => {
        expect(processSSOSettingsPayload).toHaveBeenCalled();
        expect(mockMutateAsync).toHaveBeenCalledWith(
          processedPayload,
          expect.objectContaining({
            onSuccess: expect.any(Function),
            onError: expect.any(Function),
          }),
        );
      });
    });

    it("should show success notification and calls onSuccess callback", async () => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const mockMutateAsync = vi.fn().mockImplementation((payload: any, options: any) => {
        options.onSuccess();
        return Promise.resolve({ success: true });
      });

      setupMocks({
        useSSOSettings: {
          data: createGoogleSSOData(),
          isLoading: false,
          error: null,
        },
        useEditSSOSettings: {
          mutateAsync: mockMutateAsync,
          isPending: false,
        },
      });

      const { mockOnSuccess } = renderComponent();

      await waitFor(() => {
        expect(getCapturedFormValues().sso_provider).toBe(SSO_PROVIDERS.GOOGLE);
      });

      fireEvent.click(getSaveButton());

      await waitFor(() => {
        expect(NotificationsManager.success).toHaveBeenCalledWith(
          TEST_DATA.SUCCESS_MESSAGE,
        );
        expect(mockOnSuccess).toHaveBeenCalled();
      });
    });

    it("should handle submission errors gracefully", async () => {
      const error = new Error("Submission failed");
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const mockMutateAsync = vi.fn().mockImplementation((payload: any, options: any) => {
        options.onError(error);
        return Promise.resolve();
      });

      setupMocks({
        useSSOSettings: {
          data: createGoogleSSOData(),
          isLoading: false,
          error: null,
        },
        useEditSSOSettings: {
          mutateAsync: mockMutateAsync,
          isPending: false,
        },
      });

      (parseErrorMessage as unknown as Mock).mockReturnValue("Parsed error message");

      renderComponent();

      await waitFor(() => {
        expect(getCapturedFormValues().sso_provider).toBe(SSO_PROVIDERS.GOOGLE);
      });

      fireEvent.click(getSaveButton());

      await waitFor(() => {
        expect(parseErrorMessage).toHaveBeenCalledWith(error);
        expect(NotificationsManager.fromBackend).toHaveBeenCalledWith(
          `${TEST_DATA.ERROR_MESSAGE_PREFIX} Parsed error message`,
        );
      });
    });
  });

  describe("Form Initialization", () => {
    describe("Provider Detection", () => {
      const testProviderDetection = (
        testName: string,
        ssoData: SSOData,
        expectedProvider: string,
      ) => {
        it(`should detect ${testName} provider`, async () => {
          setupMocks({
            useSSOSettings: { data: ssoData, isLoading: false, error: null },
          });

          renderComponent();

          await waitFor(() => {
            expect(getCapturedFormValues()).toMatchObject({
              sso_provider: expectedProvider,
              ...ssoData.values,
            });
          });
        });
      };

      testProviderDetection(
        "Google",
        createGoogleSSOData(),
        SSO_PROVIDERS.GOOGLE,
      );

      testProviderDetection(
        "Microsoft",
        createMicrosoftSSOData(),
        SSO_PROVIDERS.MICROSOFT,
      );

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
        SSO_PROVIDERS.OKTA,
      );

      testProviderDetection(
        "generic",
        createGenericSSOData(),
        SSO_PROVIDERS.GENERIC,
      );
    });

    describe("Role Mappings", () => {
      it("should process role mappings with all roles assigned", async () => {
        const ssoData = createRoleMappingsSSOData();

        setupMocks({
          useSSOSettings: { data: ssoData, isLoading: false, error: null },
        });

        renderComponent();

        await waitFor(() => {
          expect(getCapturedFormValues()).toMatchObject({
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

      it("should handle empty role mapping arrays", async () => {
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
          expect(getCapturedFormValues()).toMatchObject({
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
      it("should skip initialization when modal is not visible", async () => {
        const ssoData = createGoogleSSOData();

        setupMocks({
          useSSOSettings: { data: ssoData, isLoading: false, error: null },
        });

        renderComponent({ isVisible: false });

        // Dialog not rendered, no values captured
        expect(screen.queryByTestId(FORM_VALUES_TEST_ID)).not.toBeInTheDocument();
      });

      it("should skip initialization when SSO data is unavailable", async () => {
        setupMocks({
          useSSOSettings: { data: null, isLoading: false, error: null },
        });

        renderComponent();

        // Dialog renders, but form state contains no provider-derived fields.
        await waitFor(() => {
          expect(screen.getByTestId(FORM_VALUES_TEST_ID)).toBeInTheDocument();
        });
        const captured = getCapturedFormValues();
        expect(captured.sso_provider).toBeUndefined();
        expect(captured.google_client_id).toBeUndefined();
      });
    });
  });

  describe("Error Handling", () => {
    it("should handle form submission errors with undefined error message", async () => {
      const error = new Error("Network error");
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const mockMutateAsync = vi.fn().mockImplementation((payload: any, options: any) => {
        options.onError(error);
        return Promise.resolve();
      });

      (processSSOSettingsPayload as unknown as Mock).mockReturnValue({});

      setupMocks({
        useSSOSettings: {
          data: createGoogleSSOData(),
          isLoading: false,
          error: null,
        },
        useEditSSOSettings: {
          mutateAsync: mockMutateAsync,
          isPending: false,
        },
      });

      (parseErrorMessage as unknown as Mock).mockReturnValue(undefined);

      renderComponent();

      await waitFor(() => {
        expect(getCapturedFormValues().sso_provider).toBe(SSO_PROVIDERS.GOOGLE);
      });

      fireEvent.click(getSaveButton());

      await waitFor(() => {
        expect(NotificationsManager.fromBackend).toHaveBeenCalledWith(
          `${TEST_DATA.ERROR_MESSAGE_PREFIX} undefined`,
        );
      });
    });

    it("should handle form submission with malformed data", async () => {
      const mockMutateAsync = vi.fn();

      setupMocks({
        useSSOSettings: {
          data: createGoogleSSOData(),
          isLoading: false,
          error: null,
        },
        useEditSSOSettings: {
          mutateAsync: mockMutateAsync,
          isPending: false,
        },
      });

      (processSSOSettingsPayload as unknown as Mock).mockImplementation(() => {
        throw new Error("Processing failed");
      });

      renderComponent();

      await waitFor(() => {
        expect(getCapturedFormValues().sso_provider).toBe(SSO_PROVIDERS.GOOGLE);
      });

      fireEvent.click(getSaveButton());

      await waitFor(() => {
        expect(processSSOSettingsPayload).toHaveBeenCalled();
      });
      expect(mockMutateAsync).not.toHaveBeenCalled();
    });
  });

  describe("Edge Cases", () => {
    it("should handle role mappings with undefined roles object", async () => {
      const ssoData = createGoogleSSOData({
        role_mappings: {
          group_claim: "groups",
          default_role: "internal_user",
        },
      });

      setupMocks({
        useSSOSettings: { data: ssoData, isLoading: false, error: null },
      });

      renderComponent();

      await waitFor(() => {
        expect(getCapturedFormValues()).toMatchObject({
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
    it("should process team mappings when team_mappings exists", async () => {
      const ssoData = createTeamMappingsSSOData();

      setupMocks({
        useSSOSettings: { data: ssoData, isLoading: false, error: null },
      });

      renderComponent();

      await waitFor(() => {
        expect(getCapturedFormValues()).toMatchObject({
          sso_provider: SSO_PROVIDERS.GENERIC,
          ...ssoData.values,
          use_team_mappings: true,
          team_ids_jwt_field: "teams",
        });
      });
    });

    it("should handle team mappings with custom JWT field name", async () => {
      const ssoData = createTeamMappingsSSOData({
        team_ids_jwt_field: "custom_teams_field",
      });

      setupMocks({
        useSSOSettings: { data: ssoData, isLoading: false, error: null },
      });

      renderComponent();

      await waitFor(() => {
        expect(getCapturedFormValues()).toMatchObject({
          sso_provider: SSO_PROVIDERS.GENERIC,
          ...ssoData.values,
          use_team_mappings: true,
          team_ids_jwt_field: "custom_teams_field",
        });
      });
    });

    it("should handle team mappings and role mappings together", async () => {
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
        expect(getCapturedFormValues()).toMatchObject({
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

    it("should not set team mapping fields when team_mappings is not present", async () => {
      const ssoData = createGenericSSOData();

      setupMocks({
        useSSOSettings: { data: ssoData, isLoading: false, error: null },
      });

      renderComponent();

      await waitFor(() => {
        const captured = getCapturedFormValues();
        expect(captured.sso_provider).toBe(SSO_PROVIDERS.GENERIC);
        expect(captured.use_team_mappings).toBeUndefined();
        expect(captured.team_ids_jwt_field).toBeUndefined();
      });
    });

    it("should handle provider detection with partial SSO data", async () => {
      const ssoData = createSSOData({
        generic_client_id: "test-id",
        generic_authorization_endpoint: "https://unknown.provider.com/auth",
      });

      setupMocks({
        useSSOSettings: { data: ssoData, isLoading: false, error: null },
      });

      renderComponent();

      await waitFor(() => {
        expect(getCapturedFormValues()).toMatchObject({
          sso_provider: SSO_PROVIDERS.GENERIC,
          ...ssoData.values,
        });
      });
    });

    it("should handle form submission when processing throws error", async () => {
      setupMocks({
        useSSOSettings: {
          data: createGoogleSSOData(),
          isLoading: false,
          error: null,
        },
        useEditSSOSettings: { mutateAsync: vi.fn(), isPending: false },
      });

      (processSSOSettingsPayload as unknown as Mock).mockImplementation(() => {
        throw new Error("Processing error");
      });

      renderComponent();

      await waitFor(() => {
        expect(getCapturedFormValues().sso_provider).toBe(SSO_PROVIDERS.GOOGLE);
      });

      expect(() => {
        fireEvent.click(getSaveButton());
      }).not.toThrow();

      await waitFor(() => {
        expect(processSSOSettingsPayload).toHaveBeenCalled();
      });
    });
  });
});
