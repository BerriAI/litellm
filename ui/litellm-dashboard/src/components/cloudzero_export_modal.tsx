import React, { useState, useEffect } from "react";
import { Text, Button, Callout, TextInput } from "@tremor/react";
import { Modal, Form, Spin, Select } from "antd";
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
          Authorization: `Bearer ${accessToken}`,
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
        NotificationsManager.fromBackend(`Failed to load existing settings: ${errorData.error || "Unknown error"}`);
      }
    } catch (error) {
      console.error("Error loading CloudZero settings:", error);
      NotificationsManager.fromBackend("Failed to load existing settings");
    } finally {
      setSettingsLoading(false);
    }
  };

  const handleSaveCloudZeroSettings = async (values: CloudZeroSettings) => {
    if (!accessToken) {
      NotificationsManager.fromBackend("No access token available");
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
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (response.ok) {
        NotificationsManager.success(data.message || "CloudZero settings saved successfully");
        setExistingSettings({
          api_key_masked: values.api_key.substring(0, 4) + "****" + values.api_key.slice(-4),
          connection_id: values.connection_id,
          status: "configured",
        });
        return true;
      } else {
        NotificationsManager.fromBackend(data.error || "Failed to save CloudZero settings");
        return false;
      }
    } catch (error) {
      console.error("Error saving CloudZero settings:", error);
      NotificationsManager.fromBackend("Failed to save CloudZero settings");
      return false;
    } finally {
      setLoading(false);
    }
  };

  const handleExportCloudZero = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("No access token available");
      return;
    }

    setExportLoading(true);
    try {
      const response = await fetch("/cloudzero/export", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          limit: 100000,
          operation: "replace_hourly",
        }),
      });

      const data = await response.json();

      if (response.ok) {
        NotificationsManager.success(data.message || "Export to CloudZero completed successfully");
        onClose();
      } else {
        NotificationsManager.fromBackend(data.error || "Failed to export to CloudZero");
      }
    } catch (error) {
      console.error("Error exporting to CloudZero:", error);
      NotificationsManager.fromBackend("Failed to export to CloudZero");
    } finally {
      setExportLoading(false);
    }
  };

  const handleExportCSV = async () => {
    setExportLoading(true);
    try {
      // TODO: Implement CSV export functionality
      NotificationsManager.info("CSV export functionality coming soon!");
      onClose();
    } catch (error) {
      console.error("Error exporting CSV:", error);
      NotificationsManager.fromBackend("Failed to export CSV");
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
          <span>Export to CloudZero</span>
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
          <span>Export to CSV</span>
        </div>
      ),
    },
  ];

  return (
    <Modal title="Export Data" open={isOpen} onCancel={handleModalClose} footer={null} width={600} destroyOnClose>
      <div className="space-y-4">
        {/* Export Type Selection */}
        <div>
          <Text className="font-medium mb-2 block">Export Destination</Text>
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
                    title="Existing CloudZero Configuration"
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
                      API Key: {existingSettings.api_key_masked}
                      <br />
                      Connection ID: {existingSettings.connection_id}
                    </Text>
                  </Callout>
                )}

                {!existingSettings && (
                  <Form form={form} layout="vertical">
                    <Form.Item
                      label="CloudZero API Key"
                      name="api_key"
                      rules={[{ required: true, message: "Please enter your CloudZero API key" }]}
                    >
                      <TextInput type="password" placeholder="Enter your CloudZero API key" />
                    </Form.Item>

                    <Form.Item
                      label="Connection ID"
                      name="connection_id"
                      rules={[{ required: true, message: "Please enter the CloudZero connection ID" }]}
                    >
                      <TextInput placeholder="Enter CloudZero connection ID" />
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
            title="CSV Export"
            icon={() => (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
            )}
            color="blue"
          >
            <Text>Export your usage data as a CSV file for analysis in spreadsheet applications.</Text>
          </Callout>
        )}

        {/* Action Buttons */}
        <div className="flex justify-end space-x-2 pt-4">
          <Button variant="secondary" onClick={handleModalClose}>
            Cancel
          </Button>
          <Button onClick={handleExport} loading={loading || exportLoading} disabled={loading || exportLoading}>
            {exportType === "cloudzero" ? "Export to CloudZero" : "Export CSV"}
          </Button>
        </div>
      </div>
    </Modal>
  );
};

export default CloudZeroExportModal;
