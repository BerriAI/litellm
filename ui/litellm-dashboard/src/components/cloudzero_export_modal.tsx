import React, { useState, useEffect } from "react";
import { Text, Button, Callout, TextInput } from "@tremor/react";
import { Modal, Form, Spin, Select } from "antd";
import { useTranslation } from "react-i18next";
import { getGlobalLitellmHeaderName } from "@/components/networking";
import NotificationsManager from "./molecules/notifications_manager";

interface CloudZeroExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  accessToken: string | null;
}

interface CloudZeroSettings {
  api_key: string;
  connection_id: string;
}

interface CloudZeroSettingsView {
  api_key_masked: string;
  connection_id: string;
  status: string;
}

type ExportType = "cloudzero" | "csv";

const CloudZeroExportModal: React.FC<CloudZeroExportModalProps> = ({ isOpen, onClose, accessToken }) => {
  const { t } = useTranslation();
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [existingSettings, setExistingSettings] = useState<CloudZeroSettingsView | null>(null);
  const [settingsLoading, setSettingsLoading] = useState(false);
  const [exportType, setExportType] = useState<ExportType>("cloudzero");
  const [exportLoading, setExportLoading] = useState(false);

  // Load existing settings when modal opens
  useEffect(() => {
    if (isOpen && accessToken) {
      loadExistingSettings();
    }
  }, [isOpen, accessToken]);

  const loadExistingSettings = async () => {
    setSettingsLoading(true);
    try {
      const response = await fetch("/cloudzero/settings", {
        method: "GET",
        headers: {
          [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const settings = await response.json();
        setExistingSettings(settings);
        // Pre-populate form with existing settings (except masked API key)
        form.setFieldsValue({
          connection_id: settings.connection_id,
        });
      } else if (response.status !== 404) {
        // 404 means no settings configured yet, which is fine
        const errorData = await response.json();
        NotificationsManager.fromBackend(
          t("cloudzeroExportModal.failedToLoadSettings", { error: errorData.error || t("common.unknown") }),
        );
      }
    } catch (error) {
      console.error("Error loading CloudZero settings:", error);
      NotificationsManager.fromBackend(t("cloudzeroExportModal.failedToLoadSettingsGeneric"));
    } finally {
      setSettingsLoading(false);
    }
  };

  const handleSaveCloudZeroSettings = async (values: CloudZeroSettings) => {
    if (!accessToken) {
      NotificationsManager.fromBackend(t("cloudzeroExportModal.noAccessToken"));
      return;
    }

    setLoading(true);
    try {
      const endpoint = existingSettings ? "/cloudzero/settings" : "/cloudzero/init";
      const method = existingSettings ? "PUT" : "POST";

      // Add default timezone for backend compatibility
      const payload = {
        ...values,
        timezone: "UTC",
      };

      const response = await fetch(endpoint, {
        method,
        headers: {
          [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (response.ok) {
        NotificationsManager.success(data.message || t("cloudzeroExportModal.settingsSavedSuccess"));
        setExistingSettings({
          api_key_masked: values.api_key.substring(0, 4) + "****" + values.api_key.slice(-4),
          connection_id: values.connection_id,
          status: "configured",
        });
        return true;
      } else {
        NotificationsManager.fromBackend(data.error || t("cloudzeroExportModal.failedToSaveSettings"));
        return false;
      }
    } catch (error) {
      console.error("Error saving CloudZero settings:", error);
      NotificationsManager.fromBackend(t("cloudzeroExportModal.failedToSaveSettings"));
      return false;
    } finally {
      setLoading(false);
    }
  };

  const handleExportCloudZero = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend(t("cloudzeroExportModal.noAccessToken"));
      return;
    }

    setExportLoading(true);
    try {
      const response = await fetch("/cloudzero/export", {
        method: "POST",
        headers: {
          [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          limit: 100000,
          operation: "replace_hourly",
        }),
      });

      const data = await response.json();

      if (response.ok) {
        NotificationsManager.success(data.message || t("cloudzeroExportModal.exportSuccess"));
        onClose();
      } else {
        NotificationsManager.fromBackend(data.error || t("cloudzeroExportModal.failedToExport"));
      }
    } catch (error) {
      console.error("Error exporting to CloudZero:", error);
      NotificationsManager.fromBackend(t("cloudzeroExportModal.failedToExport"));
    } finally {
      setExportLoading(false);
    }
  };

  const handleExportCSV = async () => {
    setExportLoading(true);
    try {
      // TODO: Implement CSV export functionality
      NotificationsManager.info(t("cloudzeroExportModal.csvComingSoon"));
      onClose();
    } catch (error) {
      console.error("Error exporting CSV:", error);
      NotificationsManager.fromBackend(t("cloudzeroExportModal.failedToExportCsv"));
    } finally {
      setExportLoading(false);
    }
  };

  const handleExport = async () => {
    if (exportType === "cloudzero") {
      // Check if settings exist, if not save them first
      if (!existingSettings) {
        const values = await form.validateFields();
        const success = await handleSaveCloudZeroSettings(values);
        if (!success) return;
      }
      await handleExportCloudZero();
    } else {
      await handleExportCSV();
    }
  };

  const handleModalClose = () => {
    form.resetFields();
    setExportType("cloudzero");
    setExistingSettings(null);
    onClose();
  };

  const exportOptions = [
    {
      value: "cloudzero",
      label: (
        <div className="flex items-center gap-2">
          <img
            src="/cloudzero.png"
            alt="CloudZero"
            className="w-5 h-5"
            onError={(e) => {
              // Fallback to text if image fails to load
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
          <span>{t("cloudzeroExportModal.exportToCloudZero")}</span>
        </div>
      ),
    },
    {
      value: "csv",
      label: (
        <div className="flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <span>{t("cloudzeroExportModal.exportToCsv")}</span>
        </div>
      ),
    },
  ];

  return (
    <Modal
      title={t("cloudzeroExportModal.exportDataTitle")}
      open={isOpen}
      onCancel={handleModalClose}
      footer={null}
      width={600}
      destroyOnHidden
    >
      <div className="space-y-4">
        {/* Export Type Selection */}
        <div>
          <Text className="font-medium mb-2 block">{t("cloudzeroExportModal.exportDestination")}</Text>
          <Select value={exportType} onChange={setExportType} options={exportOptions} className="w-full" size="large" />
        </div>

        {/* CloudZero Configuration */}
        {exportType === "cloudzero" && (
          <div>
            {settingsLoading ? (
              <div className="flex justify-center py-8">
                <Spin size="large" />
              </div>
            ) : (
              <>
                {existingSettings && (
                  <Callout
                    title={t("cloudzeroExportModal.existingConfig")}
                    icon={() => (
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      </svg>
                    )}
                    color="green"
                    className="mb-4"
                  >
                    <Text>
                      {t("cloudzeroExportModal.apiKeyInfo", { maskedKey: existingSettings.api_key_masked })}
                      <br />
                      {t("cloudzeroExportModal.connectionIdInfo", { connectionId: existingSettings.connection_id })}
                    </Text>
                  </Callout>
                )}

                {!existingSettings && (
                  <Form form={form} layout="vertical">
                    <Form.Item
                      label={t("cloudzeroExportModal.apiKeyLabel")}
                      name="api_key"
                      rules={[{ required: true, message: t("cloudzeroExportModal.apiKeyRequired") }]}
                    >
                      <TextInput type="password" placeholder={t("cloudzeroExportModal.apiKeyPlaceholder")} />
                    </Form.Item>

                    <Form.Item
                      label={t("cloudzeroExportModal.connectionIdLabel")}
                      name="connection_id"
                      rules={[{ required: true, message: t("cloudzeroExportModal.connectionIdRequired") }]}
                    >
                      <TextInput placeholder={t("cloudzeroExportModal.connectionIdPlaceholder")} />
                    </Form.Item>
                  </Form>
                )}
              </>
            )}
          </div>
        )}

        {/* CSV Export Info */}
        {exportType === "csv" && (
          <Callout
            title={t("cloudzeroExportModal.csvExportTitle")}
            icon={() => (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
            )}
            color="blue"
          >
            <Text>{t("cloudzeroExportModal.csvExportDesc")}</Text>
          </Callout>
        )}

        {/* Action Buttons */}
        <div className="flex justify-end space-x-2 pt-4">
          <Button variant="secondary" onClick={handleModalClose}>
            {t("common.cancel")}
          </Button>
          <Button onClick={handleExport} loading={loading || exportLoading} disabled={loading || exportLoading}>
            {exportType === "cloudzero"
              ? t("cloudzeroExportModal.exportToCloudZero")
              : t("cloudzeroExportModal.exportCsvButton")}
          </Button>
        </div>
      </div>
    </Modal>
  );
};

export default CloudZeroExportModal;
