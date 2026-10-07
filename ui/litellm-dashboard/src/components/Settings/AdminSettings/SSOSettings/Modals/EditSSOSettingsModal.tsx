"use client";

import React, { useMemo } from "react";
import BaseSSOSettingsForm, {
  emptySSOSettingsFormValues,
  submitMountedSSOValues,
  useSSOSettingsForm,
  type SSOSettingsFormValues,
} from "./BaseSSOSettingsForm";
import { Button } from "@/components/ui/button";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { toast } from "@/lib/toast";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { detectSSOProvider, processSSOSettingsPayload } from "../utils";
import { useSSOSettings, type SSOSettingsValues } from "@/app/(dashboard)/hooks/sso/useSSOSettings";
import { useEditSSOSettings } from "@/app/(dashboard)/hooks/sso/useEditSSOSettings";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface EditSSOSettingsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: () => void;
}

const joinTeams = (teams: string[] | undefined): string => {
  if (!teams || teams.length === 0) return "";
  return teams.join(", ");
};

export const toSSOFormValues = (values: SSOSettingsValues): SSOSettingsFormValues => {
  const roleMappings = values.role_mappings;
  const teamMappings = values.team_mappings;

  return {
    ...emptySSOSettingsFormValues,
    sso_provider: detectSSOProvider(values) ?? "",
    google_client_id: values.google_client_id ?? "",
    google_client_secret: values.google_client_secret ?? "",
    microsoft_client_id: values.microsoft_client_id ?? "",
    microsoft_client_secret: values.microsoft_client_secret ?? "",
    microsoft_tenant: values.microsoft_tenant ?? "",
    generic_client_id: values.generic_client_id ?? "",
    generic_client_secret: values.generic_client_secret ?? "",
    generic_authorization_endpoint: values.generic_authorization_endpoint ?? "",
    generic_token_endpoint: values.generic_token_endpoint ?? "",
    generic_userinfo_endpoint: values.generic_userinfo_endpoint ?? "",
    generic_scope: values.generic_scope ?? undefined,
    saml_idp_metadata_url: values.saml_idp_metadata_url ?? undefined,
    saml_idp_metadata_xml: values.saml_idp_metadata_xml ?? undefined,
    saml_sp_entity_id: values.saml_sp_entity_id ?? undefined,
    user_email: values.user_email ?? "",
    proxy_base_url: values.proxy_base_url ?? "",
    ...(values.saml_allow_unsolicited != null
      ? { saml_allow_unsolicited: values.saml_allow_unsolicited === "true" }
      : {}),
    ...(roleMappings
      ? {
          use_role_mappings: true,
          group_claim: roleMappings.group_claim,
          default_role: roleMappings.default_role || "internal_user",
          proxy_admin_teams: joinTeams(roleMappings.roles?.proxy_admin),
          admin_viewer_teams: joinTeams(roleMappings.roles?.proxy_admin_viewer),
          internal_user_teams: joinTeams(roleMappings.roles?.internal_user),
          internal_viewer_teams: joinTeams(roleMappings.roles?.internal_user_viewer),
        }
      : {}),
    ...(teamMappings
      ? {
          use_team_mappings: true,
          team_ids_jwt_field: teamMappings.team_ids_jwt_field,
        }
      : {}),
  };
};

const EditSSOSettingsModal: React.FC<EditSSOSettingsModalProps> = ({ isVisible, onCancel, onSuccess }) => {
  const ssoSettings = useSSOSettings();
  const { mutateAsync, isPending } = useEditSSOSettings();

  const seededValues = useMemo(
    () => (ssoSettings.data?.values ? toSSOFormValues(ssoSettings.data.values) : emptySSOSettingsFormValues),
    [ssoSettings.data],
  );

  const form = useSSOSettingsForm("sso-settings", seededValues);

  const handleFormSubmit = async (formValues: SSOSettingsFormValues) => {
    try {
      const payload = processSSOSettingsPayload(formValues);

      await mutateAsync(payload, {
        onSuccess: () => {
          toast.success("SSO settings updated successfully");
          onSuccess();
        },
        onError: (error) => {
          toast.fromError("Failed to save SSO settings: " + parseErrorMessage(error));
        },
      });
    } catch (error) {
      toast.fromError("Failed to process SSO settings: " + parseErrorMessage(error));
    }
  };

  const handleCancel = () => {
    form.reset(seededValues);
    onCancel();
  };

  return (
    <Dialog open={isVisible} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[800px]">
        <DialogHeader>
          <DialogTitle>Edit SSO Settings</DialogTitle>
        </DialogHeader>
        <BaseSSOSettingsForm form={form} onFormSubmit={handleFormSubmit} />
        <DialogFooter>
          <div className="flex items-center justify-end gap-2">
            <Button type="button" variant="outline" onClick={handleCancel} disabled={isPending}>
              Cancel
            </Button>
            <Button
              type="button"
              disabled={isPending}
              onClick={submitMountedSSOValues(form, "sso-settings", handleFormSubmit)}
            >
              {isPending && <UiLoadingSpinner className="size-4 mr-1" />}
              {isPending ? "Saving..." : "Save"}
            </Button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default EditSSOSettingsModal;
