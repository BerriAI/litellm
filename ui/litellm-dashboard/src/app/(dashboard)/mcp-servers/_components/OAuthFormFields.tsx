import type { ParseKeys } from "i18next";
import { Info } from "lucide-react";
import React from "react";
import { useTranslation } from "react-i18next";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { OAUTH_FLOW } from "@/components/mcp_tools/types";
import { MountedFormField } from "@/components/common_components/MountedFormField";
import { requiredRule } from "@/components/common_components/formRules";
import TokenEndpointAuthMethodField from "./TokenEndpointAuthMethodField";
import UpstreamTokenHeaderField from "./UpstreamTokenHeaderField";
import {
  numberControl,
  parsesAsJson,
  selectControl,
  selectTriggerControl,
  tagsControl,
  textControl,
} from "./mcpFieldRules";

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

const fieldClassName = "rounded-lg border-border focus:border-info focus:ring-ring";

const OAUTH_FLOW_ITEMS: ReadonlyArray<{ value: string; labelKey: ParseKeys }> = [
  { value: OAUTH_FLOW.M2M, labelKey: "mcpServers.oauthFormFields.flowType.m2m.label" },
  { value: OAUTH_FLOW.INTERACTIVE, labelKey: "mcpServers.oauthFormFields.flowType.interactive.label" },
];

const FieldLabel: React.FC<{ label: string; tooltip: string }> = ({ label, tooltip }) => (
  <span className="text-sm font-medium text-foreground flex items-center">
    {label}
    <SimpleTooltip content={tooltip}>
      <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
    </SimpleTooltip>
  </span>
);

const UpstreamResourceField: React.FC = () => {
  const { t } = useTranslation();
  return (
    <MountedFormField
      label={
        <FieldLabel
          label={t("mcpServers.oauthFormFields.upstreamResource.label")}
          tooltip={t("mcpServers.oauthFormFields.upstreamResource.tooltip")}
        />
      }
      name={["credentials", "upstream_resource"]}
    >
      {(control) => (
        <Input
          {...textControl(control)}
          placeholder={t("mcpServers.oauthFormFields.upstreamResource.placeholder")}
          className={fieldClassName}
        />
      )}
    </MountedFormField>
  );
};

const OAuthFormFields: React.FC<OAuthFormFieldsProps> = ({
  isM2M,
  isEditing = false,
  oauthFlow,
  initialFlowType,
  docsUrl,
}) => {
  const { t } = useTranslation();
  const oauthFlowItems = OAUTH_FLOW_ITEMS.map((item) => ({ value: item.value, label: t(item.labelKey) }));
  const requiredWhenCreating = (message: string) =>
    isEditing ? undefined : { validate: { required: requiredRule(message) } };

  return (
    <>
      <MountedFormField
        label={
          <FieldLabel
            label={t("mcpServers.oauthFormFields.flowType.label")}
            tooltip={t("mcpServers.oauthFormFields.flowType.tooltip")}
          />
        }
        name="oauth_flow_type"
        {...(initialFlowType ? { defaultValue: initialFlowType } : {})}
      >
        {(control) => (
          <Select {...selectControl<string>(control)} items={oauthFlowItems}>
            <SelectTrigger {...selectTriggerControl(control)} className="w-full rounded-lg">
              <SelectValue placeholder={t("mcpServers.oauthFormFields.flowType.placeholder")} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={OAUTH_FLOW.M2M}>
                <div>
                  <span className="font-medium">{t("mcpServers.oauthFormFields.flowType.m2m.label")}</span>
                  <span className="ml-2 text-xs text-muted-foreground">
                    {t("mcpServers.oauthFormFields.flowType.m2m.description")}
                  </span>
                </div>
              </SelectItem>
              <SelectItem value={OAUTH_FLOW.INTERACTIVE}>
                <div>
                  <span className="font-medium">{t("mcpServers.oauthFormFields.flowType.interactive.label")}</span>
                  <span className="ml-2 text-xs text-muted-foreground">
                    {t("mcpServers.oauthFormFields.flowType.interactive.description")}
                  </span>
                </div>
              </SelectItem>
            </SelectContent>
          </Select>
        )}
      </MountedFormField>

      {isM2M ? (
        <>
          <MountedFormField
            label={
              <FieldLabel
                label={t("mcpServers.oauthFormFields.m2m.clientId.label")}
                tooltip={t("mcpServers.oauthFormFields.m2m.clientId.tooltip")}
              />
            }
            name={["credentials", "client_id"]}
            required={!isEditing}
            rules={requiredWhenCreating(t("mcpServers.oauthFormFields.m2m.clientId.required"))}
          >
            {(control) => (
              <PasswordInput
                {...textControl(control)}
                placeholder={
                  isEditing
                    ? t("mcpServers.oauthFormFields.m2m.clientId.placeholderEditing")
                    : t("mcpServers.oauthFormFields.m2m.clientId.placeholder")
                }
                groupClassName={fieldClassName}
              />
            )}
          </MountedFormField>
          <MountedFormField
            label={
              <FieldLabel
                label={t("mcpServers.oauthFormFields.m2m.clientSecret.label")}
                tooltip={t("mcpServers.oauthFormFields.m2m.clientSecret.tooltip")}
              />
            }
            name={["credentials", "client_secret"]}
            required={!isEditing}
            rules={requiredWhenCreating(t("mcpServers.oauthFormFields.m2m.clientSecret.required"))}
          >
            {(control) => (
              <PasswordInput
                {...textControl(control)}
                placeholder={
                  isEditing
                    ? t("mcpServers.oauthFormFields.m2m.clientSecret.placeholderEditing")
                    : t("mcpServers.oauthFormFields.m2m.clientSecret.placeholder")
                }
                groupClassName={fieldClassName}
              />
            )}
          </MountedFormField>
          <MountedFormField
            label={
              <FieldLabel
                label={t("mcpServers.oauthFormFields.m2m.tokenUrl.label")}
                tooltip={t("mcpServers.oauthFormFields.m2m.tokenUrl.tooltip")}
              />
            }
            name="token_url"
            required={!isEditing}
            rules={requiredWhenCreating(t("mcpServers.oauthFormFields.m2m.tokenUrl.required"))}
          >
            {(control) => (
              <Input
                {...textControl(control)}
                placeholder="https://auth.example.com/oauth/token"
                className={fieldClassName}
              />
            )}
          </MountedFormField>
          <TokenEndpointAuthMethodField isEditing={isEditing} />
          <MountedFormField
            label={
              <FieldLabel
                label={t("mcpServers.oauthFormFields.m2m.scopes.label")}
                tooltip={t("mcpServers.oauthFormFields.m2m.scopes.tooltip")}
              />
            }
            name={["credentials", "scopes"]}
          >
            {(control) => (
              <MultiSelect
                {...tagsControl(control, t)}
                placeholder={t("mcpServers.oauthFormFields.m2m.scopes.placeholder")}
                className="rounded-lg"
              />
            )}
          </MountedFormField>
          <UpstreamResourceField />
          <UpstreamTokenHeaderField />
        </>
      ) : (
        <>
          <MountedFormField
            label={
              <span className="flex items-center justify-between w-full">
                <FieldLabel
                  label={t("mcpServers.oauthFormFields.interactive.clientId.label")}
                  tooltip={t("mcpServers.oauthFormFields.interactive.clientId.tooltip")}
                />
                {docsUrl && (
                  <a
                    href={docsUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-info hover:text-info/80 ml-2 font-normal"
                    onClick={(e) => e.stopPropagation()}
                  >
                    {t("mcpServers.oauthFormFields.interactive.clientId.createOAuthApp")}
                  </a>
                )}
              </span>
            }
            name={["credentials", "client_id"]}
          >
            {(control) => (
              <PasswordInput
                {...textControl(control)}
                placeholder={
                  isEditing
                    ? t("mcpServers.oauthFormFields.interactive.clientId.placeholderEditing")
                    : t("mcpServers.oauthFormFields.interactive.clientId.placeholder")
                }
                groupClassName={fieldClassName}
              />
            )}
          </MountedFormField>
          <MountedFormField
            label={
              <FieldLabel
                label={t("mcpServers.oauthFormFields.interactive.clientSecret.label")}
                tooltip={t("mcpServers.oauthFormFields.interactive.clientSecret.tooltip")}
              />
            }
            name={["credentials", "client_secret"]}
          >
            {(control) => (
              <PasswordInput
                {...textControl(control)}
                placeholder={
                  isEditing
                    ? t("mcpServers.oauthFormFields.interactive.clientSecret.placeholderEditing")
                    : t("mcpServers.oauthFormFields.interactive.clientSecret.placeholder")
                }
                groupClassName={fieldClassName}
              />
            )}
          </MountedFormField>
          <MountedFormField
            label={
              <FieldLabel
                label={t("mcpServers.oauthFormFields.interactive.scopes.label")}
                tooltip={t("mcpServers.oauthFormFields.interactive.scopes.tooltip")}
              />
            }
            name={["credentials", "scopes"]}
          >
            {(control) => (
              <MultiSelect
                {...tagsControl(control, t)}
                placeholder={t("mcpServers.oauthFormFields.interactive.scopes.placeholder")}
                className="rounded-lg"
              />
            )}
          </MountedFormField>
          <UpstreamResourceField />
          <UpstreamTokenHeaderField />
          <MountedFormField
            label={
              <FieldLabel
                label={t("mcpServers.oauthFormFields.interactive.issuer.label")}
                tooltip={t("mcpServers.oauthFormFields.interactive.issuer.tooltip")}
              />
            }
            name="issuer"
          >
            {(control) => (
              <Input {...textControl(control)} placeholder="https://issuer.example.com" className={fieldClassName} />
            )}
          </MountedFormField>
          <MountedFormField
            label={
              <FieldLabel
                label={t("mcpServers.oauthFormFields.interactive.authorizationUrl.label")}
                tooltip={t("mcpServers.oauthFormFields.interactive.authorizationUrl.tooltip")}
              />
            }
            name="authorization_url"
          >
            {(control) => (
              <Input
                {...textControl(control)}
                placeholder="https://example.com/oauth/authorize"
                className={fieldClassName}
              />
            )}
          </MountedFormField>
          <MountedFormField
            label={
              <FieldLabel
                label={t("mcpServers.oauthFormFields.interactive.tokenUrl.label")}
                tooltip={t("mcpServers.oauthFormFields.interactive.tokenUrl.tooltip")}
              />
            }
            name="token_url"
          >
            {(control) => (
              <Input
                {...textControl(control)}
                placeholder="https://example.com/oauth/token"
                className={fieldClassName}
              />
            )}
          </MountedFormField>
          <TokenEndpointAuthMethodField isEditing={isEditing} />
          <MountedFormField
            label={
              <FieldLabel
                label={t("mcpServers.oauthFormFields.interactive.registrationUrl.label")}
                tooltip={t("mcpServers.oauthFormFields.interactive.registrationUrl.tooltip")}
              />
            }
            name="registration_url"
          >
            {(control) => (
              <Input
                {...textControl(control)}
                placeholder="https://example.com/oauth/register"
                className={fieldClassName}
              />
            )}
          </MountedFormField>
          <MountedFormField
            label={
              <FieldLabel
                label={t("mcpServers.oauthFormFields.interactive.tokenValidation.label")}
                tooltip={t("mcpServers.oauthFormFields.interactive.tokenValidation.tooltip")}
              />
            }
            name="token_validation_json"
            rules={{
              validate: { json: parsesAsJson(t("mcpServers.oauthFormFields.interactive.tokenValidation.invalidJson")) },
            }}
          >
            {(control) => (
              <Textarea
                {...textControl(control)}
                placeholder={'{\n  "organization": "my-org",\n  "team.id": "123"\n}'}
                rows={4}
                className="font-mono text-sm rounded-lg border-border focus:border-info focus:ring-ring"
              />
            )}
          </MountedFormField>
          <MountedFormField
            label={
              <FieldLabel
                label={t("mcpServers.oauthFormFields.interactive.tokenStorageTtl.label")}
                tooltip={t("mcpServers.oauthFormFields.interactive.tokenStorageTtl.tooltip")}
              />
            }
            name="token_storage_ttl_seconds"
          >
            {(control) => (
              <Input
                {...numberControl(control)}
                min={1}
                placeholder={t("mcpServers.oauthFormFields.interactive.tokenStorageTtl.placeholder")}
                className="w-full rounded-lg"
              />
            )}
          </MountedFormField>
          {oauthFlow && (
            <div className="rounded-lg border border-dashed border-border p-4 space-y-2">
              <p className="text-sm text-muted-foreground">
                {t("mcpServers.oauthFormFields.interactive.authorize.description")}
              </p>
              <Button
                variant="secondary"
                onClick={oauthFlow.startOAuthFlow}
                disabled={oauthFlow.status === "authorizing" || oauthFlow.status === "exchanging"}
              >
                {oauthFlow.status === "authorizing"
                  ? t("mcpServers.oauthFormFields.interactive.authorize.authorizing")
                  : oauthFlow.status === "exchanging"
                    ? t("mcpServers.oauthFormFields.interactive.authorize.exchanging")
                    : t("mcpServers.oauthFormFields.interactive.authorize.start")}
              </Button>
              {oauthFlow.error && <p className="text-sm text-destructive">{oauthFlow.error}</p>}
              {oauthFlow.status === "success" && oauthFlow.tokenResponse?.access_token && (
                <p className="text-sm text-success">
                  {t("mcpServers.oauthFormFields.interactive.authorize.success", {
                    expiresIn: oauthFlow.tokenResponse.expires_in ?? "?",
                  })}
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
