import { useCloudZeroDryRun } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroDryRun";
import { useCloudZeroExport } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroExport";
import { useCloudZeroDeleteSettings } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroSettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { Alert, Button, Card, Descriptions, Divider, Popconfirm, Tag } from "antd";
import MessageManager from "@/components/molecules/message_manager";
import { CheckCircle, Edit, Play, Trash2, Upload } from "lucide-react";
import { useState } from "react";
import CloudZeroUpdateModal from "./CloudZeroUpdateModal";
import { CloudZeroSettings } from "./types";
import { useTranslation } from "react-i18next";

interface CloudZeroIntegrationSettingsProps {
  settings: CloudZeroSettings;
  onSettingsUpdated: () => void;
}

export function CloudZeroIntegrationSettings({ settings, onSettingsUpdated }: CloudZeroIntegrationSettingsProps) {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);

  const dryRunMutation = useCloudZeroDryRun(accessToken || "");
  const exportMutation = useCloudZeroExport(accessToken || "");
  const deleteMutation = useCloudZeroDeleteSettings(accessToken || "");

  const handleDryRun = () => {
    if (!accessToken) return;

    dryRunMutation.mutate(
      { limit: 10 },
      {
        onSuccess: (data) => {
          MessageManager.success(t("cloudZero.cloudZeroIntegrationSettings.dryRunSuccess"));
        },
        onError: (error) => {
          MessageManager.error(error?.message || t("cloudZero.cloudZeroIntegrationSettings.dryRunFailed"));
        },
      },
    );
  };

  const dryRunResult = dryRunMutation.data ? JSON.stringify(dryRunMutation.data, null, 2) : null;

  const handleExport = () => {
    if (!accessToken) return;

    exportMutation.mutate(
      { operation: "replace_hourly" },
      {
        onSuccess: () => {
          MessageManager.success(t("cloudZero.cloudZeroIntegrationSettings.exportSuccess"));
        },
        onError: (error) => {
          MessageManager.error(error?.message || t("cloudZero.cloudZeroIntegrationSettings.exportFailed"));
        },
      },
    );
  };

  const handleEdit = () => {
    setIsEditModalOpen(true);
  };

  const handleEditModalOk = async () => {
    setIsEditModalOpen(false);
    onSettingsUpdated();
  };

  const handleEditModalCancel = () => {
    setIsEditModalOpen(false);
  };

  const handleDeleteClick = () => {
    setIsDeleteModalOpen(true);
  };

  const handleDeleteConfirm = () => {
    if (!accessToken) return;

    deleteMutation.mutate(undefined, {
      onSuccess: () => {
        MessageManager.success(t("cloudZero.cloudZeroIntegrationSettings.deleteSuccess"));
        setIsDeleteModalOpen(false);
        onSettingsUpdated();
      },
      onError: (error) => {
        MessageManager.error(error?.message || t("cloudZero.cloudZeroIntegrationSettings.deleteFailed"));
      },
    });
  };

  const handleDeleteCancel = () => {
    setIsDeleteModalOpen(false);
  };

  return (
    <>
      <div className="space-y-6 w-full max-w-4xl mx-auto">
        <Card
          title={
            <div className="flex items-center gap-2">
              <span className="text-lg font-semibold">{t("cloudZero.cloudZeroIntegrationSettings.cardTitle")}</span>
              <Tag color="success" className="ml-2 capitalize">
                {settings.status || t("common.active")}
              </Tag>
            </div>
          }
          extra={
            <div className="flex gap-2">
              <Button icon={<Edit size={16} />} onClick={handleEdit} className="flex items-center gap-2">
                {t("common.edit")}
              </Button>
              <Button
                danger
                icon={<Trash2 size={16} />}
                onClick={handleDeleteClick}
                className="flex items-center gap-2"
              >
                {t("common.delete")}
              </Button>
            </div>
          }
          className="shadow-sm"
        >
          <Descriptions
            bordered
            column={{
              xxl: 1,
              xl: 1,
              lg: 1,
              md: 1,
              sm: 1,
              xs: 1,
            }}
          >
            <Descriptions.Item label={t("cloudZero.cloudZeroIntegrationSettings.apiKeyRedacted")}>
              <span className="font-mono text-gray-600">
                {settings.api_key_masked || (
                  <span className="text-gray-400 italic">
                    {t("cloudZero.cloudZeroIntegrationSettings.notConfigured")}
                  </span>
                )}
              </span>
            </Descriptions.Item>
            <Descriptions.Item label={t("cloudZero.cloudZeroIntegrationSettings.connectionId")}>
              <span className="font-mono text-gray-600">
                {settings.connection_id || (
                  <span className="text-gray-400 italic">
                    {t("cloudZero.cloudZeroIntegrationSettings.notConfigured")}
                  </span>
                )}
              </span>
            </Descriptions.Item>
            <Descriptions.Item label={t("cloudZero.cloudZeroIntegrationSettings.timezone")}>
              {settings.timezone || (
                <span className="text-gray-400 italic">{t("cloudZero.cloudZeroIntegrationSettings.defaultUtc")}</span>
              )}
            </Descriptions.Item>
          </Descriptions>

          <Divider orientation="left" className="text-gray-500">
            {t("common.actions")}
          </Divider>

          <div className="flex flex-wrap gap-4 mb-6">
            <Button
              onClick={handleDryRun}
              loading={dryRunMutation.isPending}
              icon={<Play size={16} />}
              className="flex items-center gap-2"
            >
              {t("cloudZero.cloudZeroIntegrationSettings.runDryRun")}
            </Button>

            <Popconfirm
              title={t("cloudZero.cloudZeroIntegrationSettings.exportPopconfirmTitle")}
              description={t("cloudZero.cloudZeroIntegrationSettings.exportPopconfirmDescription")}
              onConfirm={handleExport}
              okText={t("cloudZero.cloudZeroIntegrationSettings.exportOkText")}
              cancelText={t("common.cancel")}
            >
              <Button
                type="primary"
                loading={exportMutation.isPending}
                icon={<Upload size={16} />}
                className="flex items-center gap-2"
              >
                {t("cloudZero.cloudZeroIntegrationSettings.exportDataNow")}
              </Button>
            </Popconfirm>
          </div>

          {dryRunResult && (
            <div className="mt-6 animate-in fade-in slide-in-from-top-4 duration-300">
              <Alert
                message={t("cloudZero.cloudZeroIntegrationSettings.dryRunResultsTitle")}
                description={
                  <div className="mt-2">
                    <p className="mb-2 text-gray-600">
                      {t("cloudZero.cloudZeroIntegrationSettings.dryRunSimulationOutput", {
                        connectionId: settings.connection_id,
                      })}
                    </p>
                    <pre className="bg-gray-50 p-4 rounded-md border border-gray-200 overflow-x-auto text-xs font-mono text-gray-800">
                      {dryRunResult}
                    </pre>
                  </div>
                }
                type="info"
                showIcon
                icon={<CheckCircle className="text-blue-500" />}
              />
            </div>
          )}
        </Card>
      </div>

      <CloudZeroUpdateModal
        open={isEditModalOpen}
        onOk={handleEditModalOk}
        onCancel={handleEditModalCancel}
        settings={settings}
      />

      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title={t("cloudZero.cloudZeroIntegrationSettings.deleteModalTitle")}
        message={t("cloudZero.cloudZeroIntegrationSettings.deleteModalMessage")}
        resourceInformationTitle={t("cloudZero.cloudZeroIntegrationSettings.integrationDetails")}
        resourceInformation={[
          {
            label: t("cloudZero.cloudZeroIntegrationSettings.connectionId"),
            value: settings.connection_id,
            code: true,
          },
          {
            label: t("cloudZero.cloudZeroIntegrationSettings.timezone"),
            value: settings.timezone || t("cloudZero.cloudZeroIntegrationSettings.defaultUtc"),
          },
        ]}
        onCancel={handleDeleteCancel}
        onOk={handleDeleteConfirm}
        confirmLoading={deleteMutation.isPending}
      />
    </>
  );
}
