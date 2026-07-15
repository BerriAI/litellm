"use client";

import { useLDAPSettings } from "@/app/(dashboard)/hooks/ldap/useLDAPSettings";
import { useUpdateLDAPSettings } from "@/app/(dashboard)/hooks/ldap/useUpdateLDAPSettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { Alert, Button, Card, Form, Input, Space, Switch, Typography } from "antd";
import { KeyRound } from "lucide-react";
import { useEffect } from "react";

const { Title, Text } = Typography;

export default function LDAPSettings() {
  const [form] = Form.useForm();
  const { accessToken } = useAuthorized();
  const { data, isLoading, isError, error, refetch } = useLDAPSettings();
  const { mutate: updateSettings, isPending } = useUpdateLDAPSettings(accessToken || "");

  useEffect(() => {
    if (!data?.values) {
      return;
    }
    form.setFieldsValue({
      ...data.values,
      ldap_bind_password: undefined,
    });
  }, [data, form]);

  const handleSubmit = (values: Record<string, unknown>) => {
    const payload: Record<string, unknown> = {
      ...values,
      ldap_enabled: Boolean(values.ldap_enabled),
      ldap_use_ssl: Boolean(values.ldap_use_ssl),
      ldap_start_tls: Boolean(values.ldap_start_tls),
      ldap_allow_insecure: Boolean(values.ldap_allow_insecure),
    };
    if (!payload.ldap_bind_password) {
      delete payload.ldap_bind_password;
    }

    updateSettings(payload, {
      onSuccess: () => {
        NotificationsManager.success("LDAP settings updated successfully");
        refetch();
      },
      onError: (updateError) => {
        NotificationsManager.fromBackend(`Failed to update LDAP settings: ${updateError.message}`);
      },
    });
  };

  const isConfigured = Boolean(data?.values.ldap_enabled && data?.values.ldap_url && data?.values.ldap_base_dn);

  return (
    <Card>
      <Space direction="vertical" size="large" className="w-full">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <KeyRound className="w-6 h-6 text-gray-400" />
            <div>
              <Title level={3}>LDAP Configuration</Title>
              <Text type="secondary">Directory login for the Admin UI</Text>
            </div>
          </div>
          <Text type={isConfigured ? "success" : "secondary"}>{isConfigured ? "Configured" : "Not configured"}</Text>
        </div>

        {isError && <Alert type="error" showIcon message={error?.message || "Failed to load LDAP settings"} />}

        <Form
          form={form}
          layout="vertical"
          onFinish={handleSubmit}
          disabled={isLoading || isPending}
          initialValues={{
            ldap_enabled: false,
            ldap_user_search_filter: "(|(uid={username})(sAMAccountName={username})(userPrincipalName={username}))",
            ldap_email_attribute: "mail",
            ldap_display_name_attribute: "displayName",
            ldap_group_attribute: "memberOf",
            ldap_use_ssl: false,
            ldap_start_tls: false,
            ldap_allow_insecure: false,
          }}
        >
          <Form.Item name="ldap_enabled" label="Enabled" valuePropName="checked">
            <Switch />
          </Form.Item>
          <Form.Item name="ldap_url" label="LDAP URL" rules={[{ required: true, message: "LDAP URL is required" }]}>
            <Input placeholder="ldap://ldap.example.com:389" />
          </Form.Item>
          <Form.Item name="ldap_base_dn" label="Base DN" rules={[{ required: true, message: "Base DN is required" }]}>
            <Input placeholder="dc=example,dc=com" />
          </Form.Item>
          <Form.Item name="ldap_search_base" label="Search Base">
            <Input placeholder="ou=People,dc=example,dc=com" />
          </Form.Item>
          <Form.Item name="ldap_bind_dn" label="Bind DN">
            <Input placeholder="cn=admin,dc=example,dc=com" />
          </Form.Item>
          <Form.Item name="ldap_bind_password" label="Bind Password">
            <Input.Password
              placeholder={data?.values.ldap_bind_password ? "Configured" : ""}
              autoComplete="new-password"
            />
          </Form.Item>
          <Form.Item
            name="ldap_user_search_filter"
            label="User Search Filter"
            rules={[{ required: true, message: "User search filter is required" }]}
          >
            <Input />
          </Form.Item>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Form.Item
              name="ldap_user_id_attribute"
              label="Stable User ID Attribute"
              extra="Use an immutable value such as objectGUID or entryUUID. Defaults to the LDAP DN."
            >
              <Input placeholder="objectGUID" />
            </Form.Item>
            <Form.Item name="ldap_email_attribute" label="Email Attribute">
              <Input />
            </Form.Item>
            <Form.Item name="ldap_display_name_attribute" label="Display Name Attribute">
              <Input />
            </Form.Item>
            <Form.Item name="ldap_group_attribute" label="Group Attribute">
              <Input />
            </Form.Item>
          </div>
          <Form.Item name="ldap_admin_group_dn" label="Admin Group DN">
            <Input placeholder="cn=litellm-admins,ou=Groups,dc=example,dc=com" />
          </Form.Item>
          <Space size="large">
            <Form.Item name="ldap_use_ssl" label="Use SSL" valuePropName="checked">
              <Switch />
            </Form.Item>
            <Form.Item name="ldap_start_tls" label="StartTLS" valuePropName="checked">
              <Switch />
            </Form.Item>
            <Form.Item name="ldap_allow_insecure" label="Allow Insecure LDAP" valuePropName="checked">
              <Switch />
            </Form.Item>
          </Space>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={isPending}>
              Save LDAP Settings
            </Button>
          </Form.Item>
        </Form>
      </Space>
    </Card>
  );
}
