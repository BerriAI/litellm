import React from "react";
import { Form, Input, InputNumber, Select, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Button, TextInput } from "@tremor/react";
import { useTranslation } from "react-i18next";
import { OAUTH_FLOW } from "./types";

interface OAuthFlowStatus {
  startOAuthFlow: () => void;
  status: string;
  error: string | null;
  tokenResponse: { access_token?: string; expires_in?: number } | null;
}

interface OAuthFormFieldsProps {
  isM2M: boolean;
  isEditing?: boolean;
  oauthFlow?: OAuthFlowStatus;
  initialFlowType?: string;
  /** Link to provider docs for creating an OAuth app (e.g. GitHub). */
  docsUrl?: string | null;
}

const fieldClassName = "rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500";

const FieldLabel: React.FC<{ label: string; tooltip: string }> = ({ label, tooltip }) => (
  <span className="text-sm font-medium text-gray-700 flex items-center">
    {label}
    <Tooltip title={tooltip}>
      <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
    </Tooltip>
  </span>
);

const OAuthFormFields: React.FC<OAuthFormFieldsProps> = ({
  isM2M,
  isEditing = false,
  oauthFlow,
  initialFlowType,
  docsUrl,
}) => {
  const { t } = useTranslation();
  const placeholderSuffix = isEditing ? ` (${t("mcpTools.oAuthFormFields.leaveBlankToKeep")})` : "";

  return (
    <>
      <Form.Item
        label={
          <FieldLabel
            label={t("mcpTools.oAuthFormFields.flowTypeLabel")}
            tooltip={t("mcpTools.oAuthFormFields.flowTypeTooltip")}
          />
        }
        name="oauth_flow_type"
        {...(initialFlowType ? { initialValue: initialFlowType } : {})}
      >
        <Select className="rounded-lg" size="large">
          <Select.Option value={OAUTH_FLOW.M2M}>
            <div>
              <span className="font-medium">{t("mcpTools.oAuthFormFields.m2mLabel")}</span>
              <span className="text-gray-400 text-xs ml-2">{t("mcpTools.oAuthFormFields.m2mDesc")}</span>
            </div>
          </Select.Option>
          <Select.Option value={OAUTH_FLOW.INTERACTIVE}>
            <div>
              <span className="font-medium">{t("mcpTools.oAuthFormFields.interactiveLabel")}</span>
              <span className="text-gray-400 text-xs ml-2">{t("mcpTools.oAuthFormFields.interactiveDesc")}</span>
            </div>
          </Select.Option>
        </Select>
      </Form.Item>

      {isM2M ? (
        <>
          <Form.Item
            label={
              <FieldLabel
                label={t("mcpTools.oAuthFormFields.clientIdLabel")}
                tooltip={t("mcpTools.oAuthFormFields.clientIdM2MTooltip")}
              />
            }
            name={["credentials", "client_id"]}
            rules={[{ required: true, message: t("mcpTools.oAuthFormFields.clientIdM2MRequired") }]}
          >
            <TextInput
              type="password"
              placeholder={`${t("mcpTools.oAuthFormFields.clientIdM2MPlaceholder")}${placeholderSuffix}`}
              className={fieldClassName}
            />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label={t("mcpTools.oAuthFormFields.clientSecretLabel")}
                tooltip={t("mcpTools.oAuthFormFields.clientSecretM2MTooltip")}
              />
            }
            name={["credentials", "client_secret"]}
            rules={[{ required: true, message: t("mcpTools.oAuthFormFields.clientSecretM2MRequired") }]}
          >
            <TextInput
              type="password"
              placeholder={`${t("mcpTools.oAuthFormFields.clientSecretM2MPlaceholder")}${placeholderSuffix}`}
              className={fieldClassName}
            />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label={t("mcpTools.oAuthFormFields.tokenUrlLabel")}
                tooltip={t("mcpTools.oAuthFormFields.tokenUrlM2MTooltip")}
              />
            }
            name="token_url"
            rules={[{ required: true, message: t("mcpTools.oAuthFormFields.tokenUrlM2MRequired") }]}
          >
            <TextInput placeholder="https://auth.example.com/oauth/token" className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label={t("mcpTools.oAuthFormFields.scopesOptionalLabel")}
                tooltip={t("mcpTools.oAuthFormFields.scopesM2MTooltip")}
              />
            }
            name={["credentials", "scopes"]}
          >
            <Select
              mode="tags"
              tokenSeparators={[","]}
              placeholder={t("mcpTools.oAuthFormFields.scopesPlaceholder")}
              className="rounded-lg"
              size="large"
            />
          </Form.Item>
        </>
      ) : (
        <>
          <Form.Item
            label={
              <span className="flex items-center justify-between w-full">
                <FieldLabel
                  label={t("mcpTools.oAuthFormFields.clientIdOptionalLabel")}
                  tooltip={t("mcpTools.oAuthFormFields.clientIdInteractiveTooltip")}
                />
                {docsUrl && (
                  <a
                    href={docsUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-blue-500 hover:text-blue-700 ml-2 font-normal"
                    onClick={(e) => e.stopPropagation()}
                  >
                    {t("mcpTools.oAuthFormFields.createOAuthAppLink")}
                  </a>
                )}
              </span>
            }
            name={["credentials", "client_id"]}
          >
            <TextInput
              type="password"
              placeholder={`${t("mcpTools.oAuthFormFields.clientIdInteractivePlaceholder")}${placeholderSuffix}`}
              className={fieldClassName}
            />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label={t("mcpTools.oAuthFormFields.clientSecretOptionalLabel")}
                tooltip={t("mcpTools.oAuthFormFields.clientSecretInteractiveTooltip")}
              />
            }
            name={["credentials", "client_secret"]}
          >
            <TextInput
              type="password"
              placeholder={`${t("mcpTools.oAuthFormFields.clientSecretInteractivePlaceholder")}${placeholderSuffix}`}
              className={fieldClassName}
            />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label={t("mcpTools.oAuthFormFields.scopesOptionalLabel")}
                tooltip={t("mcpTools.oAuthFormFields.scopesInteractiveTooltip")}
              />
            }
            name={["credentials", "scopes"]}
          >
            <Select
              mode="tags"
              tokenSeparators={[","]}
              placeholder={t("mcpTools.oAuthFormFields.scopesPlaceholder")}
              className="rounded-lg"
              size="large"
            />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label={t("mcpTools.oAuthFormFields.authorizationUrlOptionalLabel")}
                tooltip={t("mcpTools.oAuthFormFields.authorizationUrlTooltip")}
              />
            }
            name="authorization_url"
          >
            <TextInput placeholder="https://example.com/oauth/authorize" className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label={t("mcpTools.oAuthFormFields.tokenUrlOptionalLabel")}
                tooltip={t("mcpTools.oAuthFormFields.tokenUrlInteractiveTooltip")}
              />
            }
            name="token_url"
          >
            <TextInput placeholder="https://example.com/oauth/token" className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label={t("mcpTools.oAuthFormFields.registrationUrlOptionalLabel")}
                tooltip={t("mcpTools.oAuthFormFields.registrationUrlTooltip")}
              />
            }
            name="registration_url"
          >
            <TextInput placeholder="https://example.com/oauth/register" className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label={t("mcpTools.oAuthFormFields.tokenValidationRulesLabel")}
                tooltip={t("mcpTools.oAuthFormFields.tokenValidationRulesTooltip")}
              />
            }
            name="token_validation_json"
            rules={[
              {
                validator: (_: any, value: string) => {
                  if (!value || value.trim() === "") return Promise.resolve();
                  try {
                    JSON.parse(value);
                    return Promise.resolve();
                  } catch {
                    return Promise.reject(new Error(t("mcpTools.oAuthFormFields.mustBeValidJson")));
                  }
                },
              },
            ]}
          >
            <Input.TextArea
              placeholder={'{\n  "organization": "my-org",\n  "team.id": "123"\n}'}
              rows={4}
              className="font-mono text-sm rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
            />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label={t("mcpTools.oAuthFormFields.tokenStorageTtlLabel")}
                tooltip={t("mcpTools.oAuthFormFields.tokenStorageTtlTooltip")}
              />
            }
            name="token_storage_ttl_seconds"
          >
            <InputNumber
              min={1}
              placeholder={t("mcpTools.oAuthFormFields.tokenStorageTtlPlaceholder")}
              className="w-full rounded-lg"
              style={{ width: "100%" }}
            />
          </Form.Item>
          {oauthFlow && (
            <div className="rounded-lg border border-dashed border-gray-300 p-4 space-y-2">
              <p className="text-sm text-gray-600">{t("mcpTools.oAuthFormFields.oauthFlowDescription")}</p>
              <Button
                variant="secondary"
                onClick={oauthFlow.startOAuthFlow}
                disabled={oauthFlow.status === "authorizing" || oauthFlow.status === "exchanging"}
              >
                {oauthFlow.status === "authorizing"
                  ? t("mcpTools.oAuthFormFields.waitingForAuthorization")
                  : oauthFlow.status === "exchanging"
                    ? t("mcpTools.oAuthFormFields.exchangingCode")
                    : t("mcpTools.oAuthFormFields.authorizeAndFetch")}
              </Button>
              {oauthFlow.error && <p className="text-sm text-red-500">{oauthFlow.error}</p>}
              {oauthFlow.status === "success" && oauthFlow.tokenResponse?.access_token && (
                <p className="text-sm text-green-600">
                  {t("mcpTools.oAuthFormFields.tokenFetched", { seconds: oauthFlow.tokenResponse.expires_in ?? "?" })}
                </p>
              )}
            </div>
          )}
        </>
      )}
    </>
  );
};

export default OAuthFormFields;
