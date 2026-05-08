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

const LoggingSettings: React.FC = () => {
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
    const hasRetentionPeriod =
      typeof retentionPeriodValue === "string" && retentionPeriodValue.trim() !== "";

    const updateParams: StoreRequestInSpendLogsParams = {
      store_prompts_in_spend_logs: formValues.store_prompts_in_spend_logs,
      ...(hasRetentionPeriod && { maximum_spend_logs_retention_period: retentionPeriodValue }),
    };

    const submitUpdate = () =>
      mutate(updateParams, {
        onSuccess: () => NotificationsManager.success("Spend logs settings updated successfully"),
        onError: (error) =>
          NotificationsManager.fromBackend("Failed to save spend logs settings: " + parseErrorMessage(error)),
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
        onError: (deleteError) =>
          console.warn("Failed to delete retention period field (may not exist):", deleteError),
        onSettled: submitUpdate,
      },
    );
  };

  return (
    <Card title="Logging Settings">
      <Space direction="vertical" size="large" style={{ width: "100%" }}>
        <Typography.Paragraph style={{ marginBottom: 0 }} type="secondary">
          Proxy-wide settings that control how request and response data are written to spend logs.
        </Typography.Paragraph>

        <Form
          key={proxyConfigData ? JSON.stringify(initialValues) : "loading"}
          form={form}
          layout="vertical"
          onFinish={handleFormSubmit}
          initialValues={initialValues}
        >
          <Form.Item
            label="Store Prompts in Spend Logs"
            name="store_prompts_in_spend_logs"
            tooltip={
              proxyConfigData?.find((f) => f.field_name === "store_prompts_in_spend_logs")?.field_description ||
              "When enabled, prompts will be stored in spend logs for tracking and analysis purposes."
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
            label="Maximum Spend Logs Retention Period (Optional)"
            name="maximum_spend_logs_retention_period"
            tooltip={
              proxyConfigData?.find((f) => f.field_name === "maximum_spend_logs_retention_period")
                ?.field_description ||
              "Set the maximum retention period for spend logs (e.g., '7d' for 7 days, '30d' for 30 days). Leave empty for no limit."
            }
          >
            {isLoadingConfig ? (
              <Skeleton.Input active block />
            ) : (
              <Input placeholder="e.g., 7d, 30d" prefix={<ClockCircleOutlined />} />
            )}
          </Form.Item>

          <Form.Item>
            <Button
              type="primary"
              htmlType="submit"
              loading={isPending || isDeletingField}
              disabled={isLoadingConfig}
            >
              {isPending || isDeletingField ? "Saving..." : "Save Settings"}
            </Button>
          </Form.Item>
        </Form>
      </Space>
    </Card>
  );
};

export default LoggingSettings;
