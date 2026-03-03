"use client";

import { useHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useHashicorpVaultConfig";
import { useUpdateHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useUpdateHashicorpVaultConfig";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import NotificationManager from "@/components/molecules/notifications_manager";
import { Alert, Button, Card, Divider, Form, Input, Skeleton, Space, Typography } from "antd";

const SENSITIVE_FIELDS = new Set([
  "vault_token",
  "approle_role_id",
  "approle_secret_id",
  "client_key",
]);

const FIELD_LABELS: Record<string, string> = {
  vault_addr: "Vault Address",
  vault_namespace: "Namespace",
  vault_mount_name: "KV Mount Name",
  vault_path_prefix: "Path Prefix",
  vault_token: "Token",
  approle_role_id: "Role ID",
  approle_secret_id: "Secret ID",
  approle_mount_path: "Mount Path",
  client_cert: "Client Certificate",
  client_key: "Client Key",
};

interface FieldGroup {
  title: string;
  subtitle?: string;
  fields: string[];
}

const FIELD_GROUPS: FieldGroup[] = [
  {
    title: "Connection",
    fields: ["vault_addr", "vault_namespace", "vault_mount_name", "vault_path_prefix"],
  },
  {
    title: "Token Authentication",
    subtitle: "Use a Vault token to authenticate. Only one auth method is required.",
    fields: ["vault_token"],
  },
  {
    title: "AppRole Authentication",
    subtitle: "Use AppRole credentials to authenticate. Only one auth method is required.",
    fields: ["approle_role_id", "approle_secret_id", "approle_mount_path"],
  },
  {
    title: "TLS",
    subtitle: "Optional client certificate for mTLS.",
    fields: ["client_cert", "client_key"],
  },
];

export default function HashicorpVault() {
  const { accessToken } = useAuthorized();
  const { data, isLoading, isError, error } = useHashicorpVaultConfig();
  const {
    mutate: updateConfig,
    isPending: isUpdating,
  } = useUpdateHashicorpVaultConfig(accessToken);

  const schema = data?.field_schema;
  const properties = schema?.properties ?? {};
  const rawValues = data?.values ?? {};

  // Strip masked sensitive values so form inputs start empty — users enter new values or leave blank to keep existing
  const values: Record<string, any> = {};
  for (const [key, value] of Object.entries(rawValues)) {
    if (SENSITIVE_FIELDS.has(key) && typeof value === "string" && value.includes("*")) {
      // Don't populate form with masked value — it would be saved literally
      continue;
    }
    values[key] = value;
  }

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

  const renderField = (fieldName: string) => {
    const fieldSchema = properties[fieldName];
    if (!fieldSchema) return null;

    const rules =
      fieldName === "vault_addr"
        ? [{ type: "url" as const, message: "Please enter a valid URL" }]
        : undefined;

    const isSensitive = SENSITIVE_FIELDS.has(fieldName);
    const maskedValue = rawValues[fieldName];
    const hasExistingValue = isSensitive && typeof maskedValue === "string" && maskedValue.includes("*");
    const placeholder = hasExistingValue
      ? `Current: ${maskedValue}`
      : fieldSchema?.description;

    return (
      <Form.Item
        key={fieldName}
        name={fieldName}
        label={FIELD_LABELS[fieldName] ?? fieldName}
        rules={rules}
      >
        {isSensitive ? (
          <Input.Password placeholder={placeholder} />
        ) : (
          <Input placeholder={fieldSchema?.description} />
        )}
      </Form.Item>
    );
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

          <Form key={JSON.stringify(values)} layout="vertical" initialValues={values} onFinish={handleSave}>
            {FIELD_GROUPS.map((group, index) => (
              <div key={group.title}>
                {index > 0 && <Divider />}
                <Typography.Title level={5} style={{ marginBottom: 4 }}>
                  {group.title}
                </Typography.Title>
                {group.subtitle && (
                  <Typography.Paragraph type="secondary" style={{ marginBottom: 16 }}>
                    {group.subtitle}
                  </Typography.Paragraph>
                )}
                {group.fields.map(renderField)}
              </div>
            ))}

            <Divider />
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
