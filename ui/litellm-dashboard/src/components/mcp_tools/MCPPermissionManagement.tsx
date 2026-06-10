import React, { useEffect } from "react";
import { Alert, Form, Select, Tooltip, Collapse, Input, Space, Button, Switch } from "antd";
import { InfoCircleOutlined, MinusCircleOutlined, PlusOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import { MCPServer, AUTH_TYPE } from "./types";
const { Panel } = Collapse;

interface MCPPermissionManagementProps {
  availableAccessGroups: string[];
  mcpServer: MCPServer | null;
  searchValue: string;
  setSearchValue: (value: string) => void;
  getAccessGroupOptions: () => Array<{
    value: string;
    label: React.ReactNode;
  }>;
}

const MCPPermissionManagement: React.FC<MCPPermissionManagementProps> = ({
  availableAccessGroups,
  mcpServer,
  searchValue,
  setSearchValue,
  getAccessGroupOptions,
}) => {
  const { t } = useTranslation();
  const form = Form.useFormInstance();
  const watchedAuthType = Form.useWatch("auth_type", form);
  const isOAuth2 = watchedAuthType === AUTH_TYPE.OAUTH2;
  const isNoneAuth = watchedAuthType === AUTH_TYPE.NONE || watchedAuthType == null;
  const watchedExtraHeaders = Form.useWatch("extra_headers", form);
  const hasAuthorizationHeader =
    Array.isArray(watchedExtraHeaders) &&
    watchedExtraHeaders.some((h) => typeof h === "string" && h.toLowerCase() === "authorization");
  // Two distinct, independent opt-ins:
  //   - delegate_auth_to_upstream: oauth2 servers only (PKCE passthrough —
  //     bypass LiteLLM admission).
  //   - oauth_passthrough: auth_type=none + Authorization in extra_headers
  //     (OAuth pass-through: proxy upstream oauth-protected-resource, emit 401
  //     challenges, propagate upstream 401/403).
  // Kept as separate flags so neither silently implies the other and existing
  // oauth2 servers can't regress into pass-through behavior.
  const canEnableOAuthPassthrough = isNoneAuth && hasAuthorizationHeader;
  const watchedDelegateAuth = Form.useWatch("delegate_auth_to_upstream", form);
  const watchedPublicInternet = Form.useWatch("available_on_public_internet", form);
  const showInternalDelegatePkceWarning = isOAuth2 && watchedDelegateAuth === true && watchedPublicInternet === false;

  // Set initial values when mcpServer changes
  useEffect(() => {
    if (mcpServer) {
      if (mcpServer.static_headers) {
        const staticHeaders = Object.entries(mcpServer.static_headers).map(([header, value]) => ({
          header,
          value: value != null ? String(value) : "",
        }));
        form.setFieldValue("static_headers", staticHeaders);
      }
      if (Array.isArray(mcpServer.env_vars) && mcpServer.env_vars.length > 0) {
        form.setFieldValue(
          "env_vars",
          mcpServer.env_vars.map((entry) => ({
            name: entry.name,
            value: entry.value ?? "",
            scope: entry.scope ?? "global",
            description: entry.description ?? "",
          })),
        );
      }
      if (typeof mcpServer.allow_all_keys === "boolean") {
        form.setFieldValue("allow_all_keys", mcpServer.allow_all_keys);
      }
      if (typeof mcpServer.available_on_public_internet === "boolean") {
        form.setFieldValue("available_on_public_internet", mcpServer.available_on_public_internet);
      }
      if (typeof mcpServer.delegate_auth_to_upstream === "boolean") {
        form.setFieldValue("delegate_auth_to_upstream", mcpServer.delegate_auth_to_upstream);
      }
      if (typeof mcpServer.oauth_passthrough === "boolean") {
        form.setFieldValue("oauth_passthrough", mcpServer.oauth_passthrough);
      }
    } else {
      form.setFieldValue("allow_all_keys", false);
      form.setFieldValue("available_on_public_internet", true);
      form.setFieldValue("delegate_auth_to_upstream", false);
      form.setFieldValue("oauth_passthrough", false);
    }
  }, [mcpServer, form]);

  // delegate_auth_to_upstream is only honored server-side for oauth2 servers.
  // Force it back to false whenever the user switches away from oauth2 so a
  // stale toggle value doesn't get persisted unexpectedly.
  useEffect(() => {
    if (!isOAuth2) {
      form.setFieldValue("delegate_auth_to_upstream", false);
    }
  }, [isOAuth2, form]);

  // oauth_passthrough is only honored for auth_type=none servers that forward
  // Authorization upstream. Force it back to false otherwise.
  useEffect(() => {
    if (!canEnableOAuthPassthrough) {
      form.setFieldValue("oauth_passthrough", false);
    }
  }, [canEnableOAuthPassthrough, form]);

  return (
    <Collapse className="bg-gray-50 border border-gray-200 rounded-lg" expandIconPosition="end" ghost={false}>
      <Panel
        header={
          <div className="flex items-center">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
              <h3 className="text-lg font-semibold text-gray-900">
                {t("mcpTools.mcpPermissionManagement.panelTitle")}
              </h3>
            </div>
            <p className="text-sm text-gray-600 ml-4">{t("mcpTools.mcpPermissionManagement.panelSubtitle")}</p>
          </div>
        }
        key="permissions"
        className="border-0"
        forceRender
      >
        <div className="space-y-6 pt-4">
          <div className="flex items-start justify-between gap-4">
            <div>
              <span className="text-sm font-medium text-gray-700 flex items-center">
                {t("mcpTools.mcpPermissionManagement.allowAllKeysLabel")}
                <Tooltip title={t("mcpTools.mcpPermissionManagement.allowAllKeysTooltip")}>
                  <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                </Tooltip>
              </span>
              <p className="text-sm text-gray-600 mt-1">{t("mcpTools.mcpPermissionManagement.allowAllKeysDesc")}</p>
            </div>
            <Form.Item
              name="allow_all_keys"
              valuePropName="checked"
              initialValue={mcpServer?.allow_all_keys ?? false}
              className="mb-0"
            >
              <Switch />
            </Form.Item>
          </div>

          <div className="flex items-start justify-between gap-4">
            <div>
              <span className="text-sm font-medium text-gray-700 flex items-center">
                {t("mcpTools.mcpPermissionManagement.internalNetworkOnlyLabel")}
                <Tooltip title={t("mcpTools.mcpPermissionManagement.internalNetworkOnlyTooltip")}>
                  <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                </Tooltip>
              </span>
              <p className="text-sm text-gray-600 mt-1">
                {t("mcpTools.mcpPermissionManagement.internalNetworkOnlyDesc")}
              </p>
            </div>
            <Form.Item
              name="available_on_public_internet"
              valuePropName="checked"
              getValueProps={(value) => ({ checked: !value })}
              getValueFromEvent={(checked: boolean) => !checked}
              initialValue={true}
              className="mb-0"
            >
              <Switch />
            </Form.Item>
          </div>

          {isOAuth2 && (
            <div className="flex items-start justify-between gap-4">
              <div>
                <span className="text-sm font-medium text-gray-700 flex items-center">
                  {t("mcpTools.mcpPermissionManagement.delegateAuthLabel")}
                  <Tooltip title={t("mcpTools.mcpPermissionManagement.delegateAuthTooltip")}>
                    <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                  </Tooltip>
                </span>
                <p className="text-sm text-gray-600 mt-1">{t("mcpTools.mcpPermissionManagement.delegateAuthDesc")}</p>
              </div>
              <Form.Item
                name="delegate_auth_to_upstream"
                valuePropName="checked"
                initialValue={mcpServer?.delegate_auth_to_upstream ?? false}
                className="mb-0"
              >
                <Switch />
              </Form.Item>
            </div>
          )}

          {canEnableOAuthPassthrough && (
            <div className="flex items-start justify-between gap-4">
              <div>
                <span className="text-sm font-medium text-gray-700 flex items-center">
                  {t("mcpTools.mcpPermissionManagement.oauthPassthroughLabel")}
                  <Tooltip title={t("mcpTools.mcpPermissionManagement.oauthPassthroughTooltip")}>
                    <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                  </Tooltip>
                </span>
                <p className="text-sm text-gray-600 mt-1">
                  {t("mcpTools.mcpPermissionManagement.oauthPassthroughDesc")}
                </p>
              </div>
              <Form.Item
                name="oauth_passthrough"
                valuePropName="checked"
                initialValue={mcpServer?.oauth_passthrough ?? false}
                className="mb-0"
              >
                <Switch />
              </Form.Item>
            </div>
          )}

          {showInternalDelegatePkceWarning && (
            <Alert
              type="warning"
              showIcon
              className="mb-2"
              message={t("mcpTools.mcpPermissionManagement.internalDelegateWarningTitle")}
              description={t("mcpTools.mcpPermissionManagement.internalDelegateWarningDesc")}
            />
          )}

          <Form.Item
            label={
              <span className="text-sm font-medium text-gray-700 flex items-center">
                {t("mcpTools.mcpPermissionManagement.accessGroupsLabel")}
                <Tooltip title={t("mcpTools.mcpPermissionManagement.accessGroupsTooltip")}>
                  <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                </Tooltip>
              </span>
            }
            name="mcp_access_groups"
            className="mb-4"
          >
            <Select
              mode="tags"
              showSearch
              placeholder={t("mcpTools.mcpPermissionManagement.accessGroupsPlaceholder")}
              optionFilterProp="value"
              filterOption={(input, option) => (option?.value ?? "").toLowerCase().includes(input.toLowerCase())}
              onSearch={(value) => setSearchValue(value)}
              tokenSeparators={[","]}
              options={getAccessGroupOptions()}
              maxTagCount="responsive"
              allowClear
            />
          </Form.Item>

          <Form.Item
            label={
              <span className="text-sm font-medium text-gray-700 flex items-center">
                {t("mcpTools.mcpPermissionManagement.extraHeadersLabel")}
                <Tooltip title={t("mcpTools.mcpPermissionManagement.extraHeadersTooltip")}>
                  <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                </Tooltip>
                {mcpServer?.extra_headers && mcpServer.extra_headers.length > 0 && (
                  <span className="ml-2 text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">
                    {t("mcpTools.mcpPermissionManagement.configuredCount", { count: mcpServer.extra_headers.length })}
                  </span>
                )}
              </span>
            }
            name="extra_headers"
          >
            <Select
              mode="tags"
              placeholder={
                mcpServer?.extra_headers && mcpServer.extra_headers.length > 0
                  ? t("mcpTools.mcpPermissionManagement.extraHeadersCurrently", {
                      headers: mcpServer.extra_headers.join(", "),
                    })
                  : t("mcpTools.mcpPermissionManagement.extraHeadersPlaceholder")
              }
              className="rounded-lg"
              size="large"
              tokenSeparators={[","]}
              allowClear
            />
          </Form.Item>

          <Form.Item
            label={
              <span className="text-sm font-medium text-gray-700 flex items-center">
                {t("mcpTools.mcpPermissionManagement.staticHeadersLabel")}
                <Tooltip title={t("mcpTools.mcpPermissionManagement.staticHeadersTooltip")}>
                  <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                </Tooltip>
              </span>
            }
            required={false}
          >
            <Form.List name="static_headers">
              {(fields, { add, remove }) => (
                <div className="space-y-3">
                  {fields.map(({ key, name, ...restField }) => (
                    <Space key={key} className="flex w-full" align="baseline" size="middle">
                      <Form.Item
                        {...restField}
                        name={[name, "header"]}
                        className="flex-1"
                        rules={[{ required: true, message: t("mcpTools.mcpPermissionManagement.headerNameRequired") }]}
                      >
                        <Input
                          size="large"
                          allowClear
                          className="rounded-lg"
                          placeholder={t("mcpTools.mcpPermissionManagement.headerNamePlaceholder")}
                        />
                      </Form.Item>
                      <Form.Item
                        {...restField}
                        name={[name, "value"]}
                        className="flex-1"
                        rules={[{ required: true, message: t("mcpTools.mcpPermissionManagement.headerValueRequired") }]}
                      >
                        <Input
                          size="large"
                          allowClear
                          className="rounded-lg"
                          placeholder={t("mcpTools.mcpPermissionManagement.headerValuePlaceholder")}
                        />
                      </Form.Item>
                      <MinusCircleOutlined
                        onClick={() => remove(name)}
                        className="text-gray-500 hover:text-red-500 cursor-pointer"
                      />
                    </Space>
                  ))}
                  <Button type="dashed" onClick={() => add()} icon={<PlusOutlined />} block>
                    {t("mcpTools.mcpPermissionManagement.addStaticHeader")}
                  </Button>
                </div>
              )}
            </Form.List>
          </Form.Item>
        </div>
      </Panel>
    </Collapse>
  );
};

export default MCPPermissionManagement;
