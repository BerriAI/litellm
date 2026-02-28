"use client";

import { useHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useHashicorpVaultConfig";
import { useUpdateHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useUpdateHashicorpVaultConfig";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import NotificationManager from "@/components/molecules/notifications_manager";
import { Alert, Button, Card, Form, Input, Skeleton, Space, Typography } from "antd";

const SENSITIVE_FIELDS = new Set([
  "vault_token",
  "approle_role_id",
  "approle_secret_id",
  "client_key",
]);

export default function HashicorpVault() {
  const { accessToken } = useAuthorized();
  const { data, isLoading, isError, error } = useHashicorpVaultConfig();
  const {
    mutate: updateConfig,
    isPending: isUpdating,
    error: updateError,
  } = useUpdateHashicorpVaultConfig(accessToken);

  const schema = data?.field_schema;
  const properties = schema?.properties ?? {};
  const values = data?.values ?? {};

  const handleSave = (formValues: Record<string, any>) => {
    // Only send fields that have a value
    const config: Record<string, any> = {};
    for (const [key, value] of Object.entries(formValues)) {
      if (value !== undefined && value !== null && value !== "") {
        config[key] = value;
      }
    }

    updateConfig(config, {
      onSuccess: () => {
        NotificationManager.success("Hashicorp Vault configuration updated successfully");
      },
      onError: (err) => {
        NotificationManager.fromBackend(err);
      },
    });
  };

  return (
    <Card title="Hashicorp Vault">
      {isLoading ? (
        <Skeleton active />
      ) : isError ? (
        <Alert
          type="error"
          message="Could not load Hashicorp Vault configuration"
          description={error instanceof Error ? error.message : undefined}
        />
      ) : (
        <Space direction="vertical" size="large" style={{ width: "100%" }}>
          {schema?.description && (
            <Typography.Paragraph style={{ marginBottom: 0 }}>
              {schema.description}
            </Typography.Paragraph>
          )}

          {updateError && (
            <Alert
              type="error"
              message="Could not update Hashicorp Vault configuration"
              description={updateError instanceof Error ? updateError.message : undefined}
            />
          )}

          <Form layout="vertical" initialValues={values} onFinish={handleSave}>
            {Object.entries(properties).map(([fieldName, fieldSchema]: [string, any]) => (
              <Form.Item
                key={fieldName}
                name={fieldName}
                label={
                  <Typography.Text strong>
                    {fieldName}
                  </Typography.Text>
                }
                help={fieldSchema?.description}
              >
                {SENSITIVE_FIELDS.has(fieldName) ? (
                  <Input.Password placeholder={fieldName} />
                ) : (
                  <Input placeholder={fieldName} />
                )}
              </Form.Item>
            ))}

            <Form.Item>
              <Button type="primary" htmlType="submit" loading={isUpdating}>
                Save
              </Button>
            </Form.Item>
          </Form>
        </Space>
      )}
    </Card>
  );
}
