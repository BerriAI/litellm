"use client";

import { Button, Form, Modal, Space } from "antd";
import React, { useEffect } from "react";
import BaseSSOSettingsForm from "./BaseSSOSettingsForm";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { detectSSOProvider, extractRoleMappingFields, processSSOSettingsPayload } from "../utils";
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

      // Determine which SSO provider is configured
      const selectedProvider = detectSSOProvider(ssoData.values);
      const roleMappingFields = extractRoleMappingFields(ssoData.values.role_mappings);

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

      // Clear form first, then set values with a small delay to ensure proper initialization
      form.resetFields();
      setTimeout(() => {
        form.setFieldsValue(formValues);
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
