"use client";

import {
  ConfigType,
  GeneralSettingsFieldName,
  useDeleteProxyConfigField,
  useProxyConfig,
} from "@/app/(dashboard)/hooks/proxyConfig/useProxyConfig";
import {
  StoreRequestInSpendLogsParams,
  useStoreRequestInSpendLogs,
} from "@/app/(dashboard)/hooks/storeRequestInSpendLogs/useStoreRequestInSpendLogs";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { ClockCircleOutlined } from "@ant-design/icons";
import { Button, Card, Form, Input, Skeleton, Space, Switch, Typography } from "antd";
import React, { useMemo } from "react";
import { useTranslation } from "react-i18next";

const LoggingSettings: React.FC = () => {
  const { t } = useTranslation();
  const [form] = Form.useForm();
  const { mutate, isPending } = useStoreRequestInSpendLogs();
  const { mutate: deleteField, isPending: isDeletingField } = useDeleteProxyConfigField();
  const { data: proxyConfigData, isLoading: isLoadingConfig } = useProxyConfig(ConfigType.GENERAL_SETTINGS);
  const storePromptsValue = Form.useWatch("store_prompts_in_spend_logs", form);

  const initialValues = useMemo(() => {
    if (!proxyConfigData) {
      return {
        store_prompts_in_spend_logs: false,
        maximum_spend_logs_retention_period: undefined,
      };
    }

    const storePromptsField = proxyConfigData.find((field) => field.field_name === "store_prompts_in_spend_logs");
    const retentionPeriodField = proxyConfigData.find(
      (field) => field.field_name === "maximum_spend_logs_retention_period",
    );

    return {
      store_prompts_in_spend_logs: storePromptsField?.field_value ?? false,
      maximum_spend_logs_retention_period: retentionPeriodField?.field_value ?? undefined,
    };
  }, [proxyConfigData]);

  const handleFormSubmit = (formValues: StoreRequestInSpendLogsParams) => {
    const retentionPeriodValue = formValues.maximum_spend_logs_retention_period;
    const hasRetentionPeriod = typeof retentionPeriodValue === "string" && retentionPeriodValue.trim() !== "";

    const updateParams: StoreRequestInSpendLogsParams = {
      store_prompts_in_spend_logs: formValues.store_prompts_in_spend_logs,
      ...(hasRetentionPeriod && { maximum_spend_logs_retention_period: retentionPeriodValue }),
    };

    const submitUpdate = () =>
      mutate(updateParams, {
        onSuccess: () => NotificationsManager.success(t("settingsPages.loggingSettings.saveSuccess")),
        onError: (error) =>
          NotificationsManager.fromBackend(
            t("settingsPages.loggingSettings.saveFailed", { error: parseErrorMessage(error) }),
          ),
      });

    if (hasRetentionPeriod) {
      submitUpdate();
      return;
    }

    deleteField(
      {
        config_type: ConfigType.GENERAL_SETTINGS,
        field_name: GeneralSettingsFieldName.MAXIMUM_SPEND_LOGS_RETENTION_PERIOD,
      },
      {
        onError: (deleteError) => console.warn("Failed to delete retention period field (may not exist):", deleteError),
        onSettled: submitUpdate,
      },
    );
  };

  return (
    <Card title={t("settingsPages.loggingSettings.cardTitle")}>
      <Space direction="vertical" size="large" style={{ width: "100%" }}>
        <Typography.Paragraph style={{ marginBottom: 0 }} type="secondary">
          {t("settingsPages.loggingSettings.description")}
        </Typography.Paragraph>

        <Form
          key={proxyConfigData ? JSON.stringify(initialValues) : "loading"}
          form={form}
          layout="vertical"
          onFinish={handleFormSubmit}
          initialValues={initialValues}
        >
          <Form.Item
            label={t("settingsPages.loggingSettings.storePromptsLabel")}
            name="store_prompts_in_spend_logs"
            tooltip={
              proxyConfigData?.find((f) => f.field_name === "store_prompts_in_spend_logs")?.field_description ||
              t("settingsPages.loggingSettings.storePromptsTooltipDefault")
            }
            valuePropName="checked"
          >
            {isLoadingConfig ? (
              <Skeleton.Input active block />
            ) : (
              <Switch
                checked={storePromptsValue ?? false}
                onChange={(checked) => form.setFieldValue("store_prompts_in_spend_logs", checked)}
              />
            )}
          </Form.Item>

          <Form.Item
            label={t("settingsPages.loggingSettings.retentionPeriodLabel")}
            name="maximum_spend_logs_retention_period"
            tooltip={
              proxyConfigData?.find((f) => f.field_name === "maximum_spend_logs_retention_period")?.field_description ||
              t("settingsPages.loggingSettings.retentionPeriodTooltipDefault")
            }
          >
            {isLoadingConfig ? (
              <Skeleton.Input active block />
            ) : (
              <Input
                placeholder={t("settingsPages.loggingSettings.retentionPeriodPlaceholder")}
                prefix={<ClockCircleOutlined />}
              />
            )}
          </Form.Item>

          <Form.Item>
            <Button type="primary" htmlType="submit" loading={isPending || isDeletingField} disabled={isLoadingConfig}>
              {isPending || isDeletingField
                ? t("common.saving")
                : t("settingsPages.loggingSettings.saveSettingsButton")}
            </Button>
          </Form.Item>
        </Form>
      </Space>
    </Card>
  );
};

export default LoggingSettings;
