import React, { useEffect, useState } from "react";
import { FormProvider, useWatch, type UseFormReturn } from "react-hook-form";
import { Modal } from "antd";
import { Text } from "@tremor/react";
import { getSSOSettings, updateSSOSettings } from "./networking";
import { toast } from "@/lib/toast";
import { parseErrorMessage } from "./shared/errorUtils";
import { Button } from "@/components/ui/button";
import { FieldGroup } from "@/components/shared/form/field";
import {
  GroupClaimField,
  MappingToggleField,
  ProxyAdminEmailField,
  ProxyBaseUrlField,
  RoleMappingTeamFields,
  SSOProviderSelectField,
  emptySSOSettingsFormValues,
  renderProviderFields,
  submitMountedSSOValues,
  type SSOSettingsFormValues,
} from "./Settings/AdminSettings/SSOSettings/Modals/BaseSSOSettingsForm";

interface SSOModalsProps {
  isAddSSOModalVisible: boolean;
  isInstructionsModalVisible: boolean;
  handleAddSSOOk: () => void;
  handleAddSSOCancel: () => void;
  handleShowInstructions: (formValues: SSOSettingsFormValues) => void;
  handleInstructionsOk: () => void;
  handleInstructionsCancel: () => void;
  form: UseFormReturn<SSOSettingsFormValues>;
  accessToken: string | null;
  ssoConfigured?: boolean; // Add optional prop to indicate if SSO is configured
}

const detectSSOProvider = (values: Record<string, unknown>): string | null => {
  if (values.google_client_id) return "google";
  if (values.microsoft_client_id) return "microsoft";
  if (values.generic_client_id) {
    const authEndpoint =
      typeof values.generic_authorization_endpoint === "string" ? values.generic_authorization_endpoint : "";
    return authEndpoint.includes("okta") || authEndpoint.includes("auth0") ? "okta" : "generic";
  }
  if (values.saml_idp_metadata_url || values.saml_idp_metadata_xml) return "saml";
  return null;
};
const SSOModals: React.FC<SSOModalsProps> = ({
  isAddSSOModalVisible,
  isInstructionsModalVisible,
  handleAddSSOOk,
  handleAddSSOCancel,
  handleShowInstructions,
  handleInstructionsOk,
  handleInstructionsCancel,
  form,
  accessToken,
  ssoConfigured = false, // Default to false if not provided
}) => {
  const [isClearConfirmModalVisible, setIsClearConfirmModalVisible] = useState(false);
  const provider = useWatch({ control: form.control, name: "sso_provider" });
  const useRoleMappings = useWatch({ control: form.control, name: "use_role_mappings" });
  const showRoleMappingToggle = provider === "okta" || provider === "generic";

  // Load existing SSO settings when modal opens
  useEffect(() => {
    const loadSSOSettings = async () => {
      if (isAddSSOModalVisible && accessToken) {
        try {
          const ssoData = await getSSOSettings(accessToken);
          if (ssoData && ssoData.values) {
            // Determine which SSO provider is configured
            const selectedProvider = detectSSOProvider(ssoData.values);

            // Extract role mappings if they exist
            let roleMappingFields = {};
            if (ssoData.values.role_mappings) {
              const roleMappings = ssoData.values.role_mappings;

              // Helper function to join arrays into comma-separated strings
              const joinTeams = (teams: string[] | undefined): string => {
                if (!teams || teams.length === 0) return "";
                return teams.join(", ");
              };

              roleMappingFields = {
                use_role_mappings: true,
                group_claim: roleMappings.group_claim,
                default_role: roleMappings.default_role || "internal_user",
                proxy_admin_teams: joinTeams(roleMappings.roles?.proxy_admin),
                admin_viewer_teams: joinTeams(roleMappings.roles?.proxy_admin_viewer),
                internal_user_teams: joinTeams(roleMappings.roles?.internal_user),
                internal_viewer_teams: joinTeams(roleMappings.roles?.internal_user_viewer),
              };
            }

            // Set form values with existing data (excluding UI access control fields)
            const formValues: SSOSettingsFormValues = {
              sso_provider: selectedProvider ?? "",
              proxy_base_url: ssoData.values.proxy_base_url,
              user_email: ssoData.values.user_email,
              google_client_id: ssoData.values.google_client_id,
              google_client_secret: ssoData.values.google_client_secret,
              microsoft_client_id: ssoData.values.microsoft_client_id,
              microsoft_client_secret: ssoData.values.microsoft_client_secret,
              microsoft_tenant: ssoData.values.microsoft_tenant,
              generic_client_id: ssoData.values.generic_client_id,
              generic_client_secret: ssoData.values.generic_client_secret,
              generic_authorization_endpoint: ssoData.values.generic_authorization_endpoint,
              generic_token_endpoint: ssoData.values.generic_token_endpoint,
              generic_userinfo_endpoint: ssoData.values.generic_userinfo_endpoint,
              generic_scope: ssoData.values.generic_scope,
              saml_idp_metadata_url: ssoData.values.saml_idp_metadata_url,
              saml_idp_metadata_xml: ssoData.values.saml_idp_metadata_xml,
              saml_sp_entity_id: ssoData.values.saml_sp_entity_id,
              ...roleMappingFields,
              saml_allow_unsolicited: ssoData.values.saml_allow_unsolicited === "true",
            };

            form.reset({ ...emptySSOSettingsFormValues, ...formValues });
          }
        } catch (error) {
          console.error("Failed to load SSO settings:", error);
        }
      }
    };

    loadSSOSettings();
  }, [isAddSSOModalVisible, accessToken, form]);

  // Enhanced form submission handler
  const handleFormSubmit = async (formValues: SSOSettingsFormValues) => {
    if (!accessToken) {
      toast.fromError("No access token available");
      return;
    }

    try {
      const {
        proxy_admin_teams,
        admin_viewer_teams,
        internal_user_teams,
        internal_viewer_teams,
        default_role,
        group_claim,
        use_role_mappings,
        ...rest
      } = formValues;

      const payload: Record<string, unknown> = {
        ...rest,
      };

      if (typeof payload.saml_allow_unsolicited === "boolean") {
        payload.saml_allow_unsolicited = payload.saml_allow_unsolicited ? "true" : "false";
      }

      // Add role mappings if use_role_mappings is checked
      if (use_role_mappings) {
        // Helper function to split comma-separated string into array
        const splitTeams = (teams: string | undefined): string[] => {
          if (!teams || teams.trim() === "") return [];
          return teams
            .split(",")
            .map((team) => team.trim())
            .filter((team) => team.length > 0);
        };

        // Map default role display values to backend values
        const defaultRoleMapping: Record<string, string> = {
          internal_user_viewer: "internal_user_viewer",
          internal_user: "internal_user",
          proxy_admin_viewer: "proxy_admin_viewer",
          proxy_admin: "proxy_admin",
        };

        payload.role_mappings = {
          provider: "generic",
          group_claim,
          default_role: (default_role ? defaultRoleMapping[default_role] : undefined) || "internal_user",
          roles: {
            proxy_admin: splitTeams(proxy_admin_teams),
            proxy_admin_viewer: splitTeams(admin_viewer_teams),
            internal_user: splitTeams(internal_user_teams),
            internal_user_viewer: splitTeams(internal_viewer_teams),
          },
        };
      }

      // Save SSO settings using the new API
      await updateSSOSettings(accessToken, payload);

      // Continue with the original flow (show instructions)
      handleShowInstructions(formValues);
    } catch (error: unknown) {
      toast.fromError("Failed to save SSO settings: " + parseErrorMessage(error));
    }
  };

  // Handle clearing SSO settings
  const handleClearSSO = async () => {
    if (!accessToken) {
      toast.fromError("No access token available");
      return;
    }

    try {
      // Clear all SSO settings
      const clearSettings = {
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
      };

      await updateSSOSettings(accessToken, clearSettings);

      // Clear the form
      form.reset(emptySSOSettingsFormValues);

      // Close the confirmation modal
      setIsClearConfirmModalVisible(false);

      // Close the main SSO modal and trigger refresh
      handleAddSSOOk();

      toast.success("SSO settings cleared successfully");
    } catch (error) {
      console.error("Failed to clear SSO settings:", error);
      toast.fromError("Failed to clear SSO settings");
    }
  };

  // Helper function to render provider fields
  return (
    <>
      <Modal
        title={ssoConfigured ? "Edit SSO Settings" : "Add SSO"}
        open={isAddSSOModalVisible}
        width={800}
        footer={null}
        onOk={handleAddSSOOk}
        onCancel={handleAddSSOCancel}
      >
        <FormProvider {...form}>
          <form
            onSubmit={(event) => {
              event.preventDefault();
              submitMountedSSOValues(form, "admin-panel", handleFormSubmit)();
            }}
          >
            <FieldGroup>
              <SSOProviderSelectField />
              {provider ? renderProviderFields(provider) : null}
              <ProxyAdminEmailField />
              <ProxyBaseUrlField />
              {showRoleMappingToggle && <MappingToggleField name="use_role_mappings" label="Use Role Mappings" />}
              {useRoleMappings && (
                <>
                  <GroupClaimField />
                  <RoleMappingTeamFields />
                </>
              )}
            </FieldGroup>
            <div className="mt-4 flex items-center justify-end gap-2">
              {ssoConfigured && (
                <Button type="button" variant="secondary" onClick={() => setIsClearConfirmModalVisible(true)}>
                  Clear
                </Button>
              )}
              <Button type="submit">Save</Button>
            </div>
          </form>
        </FormProvider>
      </Modal>

      {/* Clear Confirmation Modal */}
      <Modal
        title="Confirm Clear SSO Settings"
        open={isClearConfirmModalVisible}
        onOk={handleClearSSO}
        onCancel={() => setIsClearConfirmModalVisible(false)}
        okText="Yes, Clear"
        cancelText="Cancel"
        okButtonProps={{
          danger: true,
          style: {
            backgroundColor: "#dc2626",
            borderColor: "#dc2626",
          },
        }}
      >
        <p>Are you sure you want to clear all SSO settings? This action cannot be undone.</p>
        <p>Users will no longer be able to login using SSO after this change.</p>
      </Modal>

      <Modal
        title="SSO Setup Instructions"
        open={isInstructionsModalVisible}
        width={800}
        footer={null}
        onOk={handleInstructionsOk}
        onCancel={handleInstructionsCancel}
      >
        <p>Follow these steps to complete the SSO setup:</p>
        <Text className="mt-2">1. DO NOT Exit this TAB</Text>
        <Text className="mt-2">2. Open a new tab, visit your proxy base url</Text>
        <Text className="mt-2">3. Confirm your SSO is configured correctly and you can login on the new Tab</Text>
        <Text className="mt-2">4. If Step 3 is successful, you can close this tab</Text>
        <div style={{ textAlign: "right", marginTop: "10px" }}>
          <Button type="button" onClick={handleInstructionsOk}>
            Done
          </Button>
        </div>
      </Modal>
    </>
  );
};

export default SSOModals;
