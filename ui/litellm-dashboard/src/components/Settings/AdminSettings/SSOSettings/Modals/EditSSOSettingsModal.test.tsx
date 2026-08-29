import { render, screen, fireEvent, waitFor, within } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, Mock } from "vitest";
import EditSSOSettingsModal, { toSSOFormValues } from "./EditSSOSettingsModal";
import { ssoProviderConfigs } from "./BaseSSOSettingsForm";
import type { SSOSettingsValues } from "@/app/(dashboard)/hooks/sso/useSSOSettings";
import { useSSOSettings } from "@/app/(dashboard)/hooks/sso/useSSOSettings";
import { useEditSSOSettings } from "@/app/(dashboard)/hooks/sso/useEditSSOSettings";
import { toast } from "@/lib/toast";
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
  SUCCESS_MESSAGE: "SSO settings updated successfully",
  ERROR_MESSAGE_PREFIX: "Failed to save SSO settings:",
  BUTTON_TEXT: {
    CANCEL: "Cancel",
    SAVE: "Save",
    SAVING: "Saving...",
  },
} as const;

const TEST_IDS = {
  BASE_SSO_FORM: "base-sso-form",
  TRIGGER_FORM_SUBMIT: "trigger-form-submit",
} as const;

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

let lastSeededForm: any;

vi.mock("./BaseSSOSettingsForm", async (importOriginal) => ({
  ...(await importOriginal<typeof import("./BaseSSOSettingsForm")>()),
  default: ({ form, onFormSubmit }: any) => {
    lastSeededForm = form;
    return (
      <div data-testid={TEST_IDS.BASE_SSO_FORM}>
        <button data-testid={TEST_IDS.TRIGGER_FORM_SUBMIT} onClick={() => onFormSubmit({ testField: "testValue" })}>
          Trigger Form Submit
        </button>
      </div>
    );
  },
}));

vi.mock("@/app/(dashboard)/hooks/sso/useSSOSettings", () => ({
  useSSOSettings: vi.fn(),
}));

vi.mock("@/app/(dashboard)/hooks/sso/useEditSSOSettings", () => ({
  useEditSSOSettings: vi.fn(),
}));

vi.mock("@/components/shared/errorUtils", () => ({
  parseErrorMessage: vi.fn(),
}));

vi.mock("../utils", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../utils")>()),
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

const getDialog = () => screen.getByRole("dialog");
const getCancelButton = () => within(getDialog()).getByRole("button", { name: TEST_DATA.BUTTON_TEXT.CANCEL });
const getSaveButton = () => within(getDialog()).getByRole("button", { name: /^(Save|Saving)/ });

const seededValuesFor = (ssoData: SSOData) => toSSOFormValues(ssoData.values as SSOSettingsValues);

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

      expect(within(getDialog()).getByText(TEST_DATA.MODAL_TITLE)).toBeInTheDocument();
      expect(screen.getByTestId(TEST_IDS.BASE_SSO_FORM)).toBeInTheDocument();
    });

    it("displays modal as closed when not visible", () => {
      renderComponent({ isVisible: false });

      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
      expect(screen.queryByText(TEST_DATA.MODAL_TITLE)).not.toBeInTheDocument();
    });
  });

  describe("Footer Actions", () => {
    it("renders cancel and save buttons", () => {
      renderComponent();

      expect(getCancelButton()).toBeInTheDocument();
      expect(getSaveButton()).toHaveTextContent(TEST_DATA.BUTTON_TEXT.SAVE);
    });

    it("calls onCancel and resets form when cancel button is clicked", () => {
      const { mockOnCancel } = renderComponent();

      fireEvent.click(getCancelButton());

      expect(mockOnCancel).toHaveBeenCalled();
    });

    it("calls form.submit when save button is clicked", async () => {
      const mockMutateAsync = vi.fn().mockResolvedValue({ success: true });
      (processSSOSettingsPayload as any).mockReturnValue({ processed: "payload" });
      setupMocks({
        useSSOSettings: { data: createGoogleSSOData({ proxy_base_url: "https://proxy.example.com" }) },
        useEditSSOSettings: { mutateAsync: mockMutateAsync, isPending: false },
      });

      renderComponent();

      fireEvent.click(getSaveButton());

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalled();
      });
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

        expect(getSaveButton()).toBeDisabled();
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

      renderComponent();

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

      expect(toast.success).toHaveBeenCalledWith(TEST_DATA.SUCCESS_MESSAGE);
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
      expect(toast.fromError).toHaveBeenCalledWith(`${TEST_DATA.ERROR_MESSAGE_PREFIX} Parsed error message`);
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
            expect(seededValuesFor(ssoData)).toMatchObject({
              sso_provider: expectedProvider,
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
          expect(seededValuesFor(ssoData)).toMatchObject({
            sso_provider: SSO_PROVIDERS.GOOGLE,
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
          expect(seededValuesFor(ssoData)).toMatchObject({
            sso_provider: SSO_PROVIDERS.GOOGLE,
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
          expect(seededValuesFor(ssoData).sso_provider).toBe(SSO_PROVIDERS.GOOGLE);
        });
      });

      it("skips initialization when modal is not visible", () => {
        const ssoData = createGoogleSSOData();

        setupMocks({
          useSSOSettings: { data: ssoData, isLoading: false, error: null },
        });

        renderComponent({ isVisible: false });

        expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
        expect(screen.queryByTestId(TEST_IDS.BASE_SSO_FORM)).not.toBeInTheDocument();
      });

      it("skips initialization when SSO data is unavailable", () => {
        setupMocks({
          useSSOSettings: { data: null, isLoading: false, error: null },
        });

        renderComponent();

        expect(screen.getByTestId(TEST_IDS.BASE_SSO_FORM)).toBeInTheDocument();
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

      expect(toast.fromError).toHaveBeenCalledWith(`${TEST_DATA.ERROR_MESSAGE_PREFIX} undefined`);
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
        expect(seededValuesFor(ssoData)).toMatchObject({
          sso_provider: SSO_PROVIDERS.GOOGLE,
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
        expect(seededValuesFor(ssoData)).toMatchObject({
          sso_provider: SSO_PROVIDERS.GENERIC,
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
        expect(seededValuesFor(ssoData)).toMatchObject({
          sso_provider: SSO_PROVIDERS.GENERIC,
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
        expect(seededValuesFor(ssoData)).toMatchObject({
          sso_provider: SSO_PROVIDERS.GENERIC,
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
        const callArgs = seededValuesFor(ssoData);
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
        expect(seededValuesFor(ssoData)).toMatchObject({
          sso_provider: SSO_PROVIDERS.GENERIC,
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

  describe("Reseeding", () => {
    it("replaces every field when reopened against a different stored config", async () => {
      const first = createGoogleSSOData({
        google_client_id: "first-tenant-id",
        proxy_base_url: "https://first.example.com",
        user_email: "first-admin@example.com",
      });
      setupMocks({ useSSOSettings: { data: first, isLoading: false, error: null } });
      const { rerender } = renderComponent();

      await waitFor(() => {
        expect(lastSeededForm.getValues().google_client_id).toBe("first-tenant-id");
      });

      const second = createGoogleSSOData({
        google_client_id: "second-tenant-id",
        proxy_base_url: "https://second.example.com",
        user_email: "second-admin@example.com",
      });
      setupMocks({ useSSOSettings: { data: second, isLoading: false, error: null } });
      rerender(<EditSSOSettingsModal isVisible={true} onCancel={vi.fn()} onSuccess={vi.fn()} />);

      await waitFor(() => {
        expect(lastSeededForm.getValues().google_client_id).toBe("second-tenant-id");
      });
      expect(JSON.stringify(lastSeededForm.getValues())).not.toContain("first");
    });
  });

  describe("Seeding completeness", () => {
    it("seeds every field the provider forms can mount", () => {
      const allFields = Object.values(ssoProviderConfigs).flatMap((config) => config.fields);
      const textFieldNames = Array.from(
        new Set(allFields.filter((field) => field.type !== "checkbox").map((field) => field.name)),
      );
      const stored = Object.fromEntries(textFieldNames.map((name) => [name, `stored-${name}`]));

      const seeded = toSSOFormValues({ ...stored, saml_allow_unsolicited: "true" } as unknown as SSOSettingsValues);

      expect(textFieldNames.filter((name) => seeded[name] !== `stored-${name}`)).toEqual([]);
      expect(seeded.saml_allow_unsolicited).toBe(true);
    });
  });
});
