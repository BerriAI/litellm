import React, { useEffect, useState } from "react";
import { FormProvider, useWatch, type UseFormReturn } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { getSSOSettings, updateSSOSettings } from "./networking";
import { toast } from "@/lib/toast";
import { parseErrorMessage } from "./shared/errorUtils";
import { Button } from "@/components/ui/button";
import { FieldGroup } from "@/components/ui/field";
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
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

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
  const { t } = useTranslation();
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
      toast.fromError(t("adminPanel.ssoModals.toasts.noAccessToken"));
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
      toast.fromError(t("adminPanel.ssoModals.toasts.saveFailed", { error: parseErrorMessage(error) }));
    }
  };

  // Handle clearing SSO settings
  const handleClearSSO = async () => {
    if (!accessToken) {
      toast.fromError(t("adminPanel.ssoModals.toasts.noAccessToken"));
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

      toast.success(t("adminPanel.ssoModals.toasts.clearSuccess"));
    } catch (error) {
      console.error("Failed to clear SSO settings:", error);
      toast.fromError(t("adminPanel.ssoModals.toasts.clearFailed"));
    }
  };

  // Helper function to render provider fields
  return (
    <>
      <Dialog open={isAddSSOModalVisible} onOpenChange={(open) => !open && handleAddSSOCancel()}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[800px]">
          <DialogHeader>
            <DialogTitle>
              {ssoConfigured ? t("adminPanel.ssoModals.titleEdit") : t("adminPanel.ssoModals.titleAdd")}
            </DialogTitle>
          </DialogHeader>
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
                {showRoleMappingToggle && (
                  <MappingToggleField name="use_role_mappings" label={t("adminPanel.ssoModals.useRoleMappings")} />
                )}
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
                    {t("adminPanel.ssoModals.clear")}
                  </Button>
                )}
                <Button type="submit">{t("adminPanel.ssoModals.save")}</Button>
              </div>
            </form>
          </FormProvider>
        </DialogContent>
      </Dialog>

      {/* Clear Confirmation Modal */}
      <Dialog open={isClearConfirmModalVisible} onOpenChange={(open) => !open && setIsClearConfirmModalVisible(false)}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{t("adminPanel.ssoModals.clearConfirm.title")}</DialogTitle>
          </DialogHeader>
          <p>{t("adminPanel.ssoModals.clearConfirm.message")}</p>
          <p>{t("adminPanel.ssoModals.clearConfirm.warning")}</p>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsClearConfirmModalVisible(false)}>
              {t("adminPanel.ssoModals.clearConfirm.cancel")}
            </Button>
            <Button onClick={handleClearSSO} variant="destructive">
              {t("adminPanel.ssoModals.clearConfirm.confirm")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={isInstructionsModalVisible} onOpenChange={(open) => !open && handleInstructionsCancel()}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[800px]">
          <DialogHeader>
            <DialogTitle>{t("adminPanel.ssoModals.instructions.title")}</DialogTitle>
          </DialogHeader>
          <p>{t("adminPanel.ssoModals.instructions.intro")}</p>
          <p className="text-sm mt-2">{t("adminPanel.ssoModals.instructions.step1")}</p>
          <p className="text-sm mt-2">{t("adminPanel.ssoModals.instructions.step2")}</p>
          <p className="text-sm mt-2">{t("adminPanel.ssoModals.instructions.step3")}</p>
          <p className="text-sm mt-2">{t("adminPanel.ssoModals.instructions.step4")}</p>
          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button type="button" onClick={handleInstructionsOk}>
              {t("adminPanel.ssoModals.instructions.done")}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
};

export default SSOModals;
