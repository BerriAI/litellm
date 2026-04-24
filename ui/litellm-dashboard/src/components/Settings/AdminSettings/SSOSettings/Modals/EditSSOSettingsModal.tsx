"use client";

import React, { useEffect } from "react";
import { FormProvider, useForm } from "react-hook-form";

import NotificationsManager from "@/components/molecules/notifications_manager";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useEditSSOSettings } from "@/app/(dashboard)/hooks/sso/useEditSSOSettings";
import { useSSOSettings } from "@/app/(dashboard)/hooks/sso/useSSOSettings";

import BaseSSOSettingsForm, { SSOSettingsFormValues } from "./BaseSSOSettingsForm";
import { processSSOSettingsPayload } from "../utils";

interface EditSSOSettingsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: () => void;
}

const emptyValues: SSOSettingsFormValues = {};

const EditSSOSettingsModal: React.FC<EditSSOSettingsModalProps> = ({
  isVisible,
  onCancel,
  onSuccess,
}) => {
  const form = useForm<SSOSettingsFormValues>({
    defaultValues: emptyValues,
    mode: "onSubmit",
  });

  const ssoSettings = useSSOSettings();
  const { mutateAsync, isPending } = useEditSSOSettings();

  useEffect(() => {
    if (isVisible && ssoSettings.data && ssoSettings.data.values) {
      const ssoData = ssoSettings.data;

      let selectedProvider: string | undefined = undefined;
      if (ssoData.values.google_client_id) {
        selectedProvider = "google";
      } else if (ssoData.values.microsoft_client_id) {
        selectedProvider = "microsoft";
      } else if (ssoData.values.generic_client_id) {
        if (
          ssoData.values.generic_authorization_endpoint?.includes("okta") ||
          ssoData.values.generic_authorization_endpoint?.includes("auth0")
        ) {
          selectedProvider = "okta";
        } else {
          selectedProvider = "generic";
        }
      }

      let roleMappingFields: Partial<SSOSettingsFormValues> = {};
      if (ssoData.values.role_mappings) {
        const roleMappings = ssoData.values.role_mappings;

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

      let teamMappingFields: Partial<SSOSettingsFormValues> = {};
      if (ssoData.values.team_mappings) {
        const teamMappings = ssoData.values.team_mappings;
        teamMappingFields = {
          use_team_mappings: true,
          team_ids_jwt_field: teamMappings.team_ids_jwt_field,
        };
      }

      const formValues: SSOSettingsFormValues = {
        sso_provider: selectedProvider,
        ...ssoData.values,
        ...roleMappingFields,
        ...teamMappingFields,
      };

      form.reset(formValues);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isVisible, ssoSettings.data]);

  const onSubmit = form.handleSubmit(async (formValues) => {
    try {
      const payload = processSSOSettingsPayload(formValues);

      await mutateAsync(payload, {
        onSuccess: () => {
          NotificationsManager.success("SSO settings updated successfully");
          onSuccess();
        },
        onError: (error) => {
          NotificationsManager.fromBackend(
            "Failed to save SSO settings: " + parseErrorMessage(error),
          );
        },
      });
    } catch (error) {
      NotificationsManager.fromBackend(
        "Failed to process SSO settings: " + parseErrorMessage(error),
      );
    }
  });

  const handleCancel = () => {
    form.reset(emptyValues);
    onCancel();
  };

  return (
    <Dialog
      open={isVisible}
      onOpenChange={(o) => (!o ? handleCancel() : undefined)}
    >
      <DialogContent className="max-w-[800px]">
        <DialogHeader>
          <DialogTitle>Edit SSO Settings</DialogTitle>
        </DialogHeader>
        <FormProvider {...form}>
          <form onSubmit={onSubmit}>
            <BaseSSOSettingsForm />
            <DialogFooter className="mt-6">
              <Button
                type="button"
                variant="outline"
                onClick={handleCancel}
                disabled={isPending}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={isPending}>
                {isPending ? "Saving..." : "Save"}
              </Button>
            </DialogFooter>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
};

export default EditSSOSettingsModal;
