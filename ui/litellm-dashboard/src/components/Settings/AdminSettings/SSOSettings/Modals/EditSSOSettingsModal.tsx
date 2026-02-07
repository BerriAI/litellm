"use client";

import { Button, Form, Modal, Space } from "antd";
import React, { useEffect } from "react";
import BaseSSOSettingsForm from "./BaseSSOSettingsForm";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { processSSOSettingsPayload } from "../utils";
import { useSSOSettings } from "@/app/(dashboard)/hooks/sso/useSSOSettings";
import { useEditSSOSettings } from "@/app/(dashboard)/hooks/sso/useEditSSOSettings";

interface EditSSOSettingsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: () => void;
}

const EditSSOSettingsModal: React.FC<EditSSOSettingsModalProps> = ({ isVisible, onCancel, onSuccess }) => {
  const [form] = Form.useForm();

  // Use react-query hooks for SSO settings
  const ssoSettings = useSSOSettings();
  const { mutateAsync, isPending } = useEditSSOSettings();
  useEffect(() => {
    if (isVisible && ssoSettings.data && ssoSettings.data.values) {
      const ssoData = ssoSettings.data;
      console.log("Raw SSO data received:", ssoData); // Debug log
      console.log("SSO values:", ssoData.values); // Debug log
      console.log("user_email from API:", ssoData.values.user_email); // Debug log

      // Determine which SSO provider is configured
      let selectedProvider = null;
      if (ssoData.values.google_client_id) {
        selectedProvider = "google";
      } else if (ssoData.values.microsoft_client_id) {
        selectedProvider = "microsoft";
      } else if (ssoData.values.generic_client_id) {
        // Check if it looks like Okta based on endpoints
        if (
          ssoData.values.generic_authorization_endpoint?.includes("okta") ||
          ssoData.values.generic_authorization_endpoint?.includes("auth0")
        ) {
          selectedProvider = "okta";
        } else {
          selectedProvider = "generic";
        }
      }

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

      // Extract team mappings if they exist
      let teamMappingFields = {};
      if (ssoData.values.team_mappings) {
        const teamMappings = ssoData.values.team_mappings;
        teamMappingFields = {
          use_team_mappings: true,
          team_ids_jwt_field: teamMappings.team_ids_jwt_field,
        };
      }

      // Set form values with existing data (excluding UI access control fields)
      const formValues = {
        sso_provider: selectedProvider,
        ...ssoData.values,
        ...roleMappingFields,
        ...teamMappingFields,
      };

      console.log("Setting form values:", formValues); // Debug log

      // Clear form first, then set values with a small delay to ensure proper initialization
      form.resetFields();
      setTimeout(() => {
        form.setFieldsValue(formValues);
        console.log("Form values set, current form values:", form.getFieldsValue()); // Debug log
      }, 100);
    }
  }, [isVisible, ssoSettings.data, form]);

  // Enhanced form submission handler
  const handleFormSubmit = async (formValues: Record<string, any>) => {
    try {
      const payload = processSSOSettingsPayload(formValues);

      await mutateAsync(payload, {
        onSuccess: () => {
          NotificationsManager.success("SSO settings updated successfully");
          onSuccess();
        },
        onError: (error) => {
          NotificationsManager.fromBackend("Failed to save SSO settings: " + parseErrorMessage(error));
        },
      });
    } catch (error) {
      // Handle processing errors gracefully
      NotificationsManager.fromBackend("Failed to process SSO settings: " + parseErrorMessage(error));
    }
  };

  const handleCancel = () => {
    form.resetFields();
    onCancel();
  };

  return (
    <Modal
      title="Edit SSO Settings"
      open={isVisible}
      width={800}
      footer={
        <Space>
          <Button onClick={handleCancel} disabled={isPending}>
            Cancel
          </Button>
          <Button loading={isPending} onClick={() => form.submit()}>
            {isPending ? "Saving..." : "Save"}
          </Button>
        </Space>
      }
      onCancel={handleCancel}
    >
      <BaseSSOSettingsForm form={form} onFormSubmit={handleFormSubmit} />
    </Modal>
  );
};

export default EditSSOSettingsModal;
