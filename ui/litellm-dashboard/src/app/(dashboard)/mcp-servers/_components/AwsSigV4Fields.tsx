import { Info } from "lucide-react";
import React from "react";
import { useTranslation } from "react-i18next";
import { SimpleTooltip } from "@/components/ui/tooltip";

import { MountedFormField } from "@/components/common_components/MountedFormField";
import { requiredRule } from "@/components/common_components/formRules";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { Input } from "@/components/ui/input";
import { requiredWhenSiblingSet, textControl } from "./mcpFieldRules";

const fieldClassName = "rounded-lg border-border focus:border-info focus:ring-ring";

const FieldLabel: React.FC<{ label: string; tooltip: string }> = ({ label, tooltip }) => (
  <span className="text-sm font-medium text-foreground flex items-center">
    {label}
    <SimpleTooltip content={tooltip}>
      <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
    </SimpleTooltip>
  </span>
);

const ACCESS_KEY_PATH = ["credentials", "aws_access_key_id"] as const;
const SECRET_KEY_PATH = ["credentials", "aws_secret_access_key"] as const;

const AwsSigV4Fields: React.FC = () => {
  const { t } = useTranslation();
  return (
    <>
      <p className="text-sm text-muted-foreground mb-2">
        {t("mcpServers.awsSigV4Fields.intro")}{" "}
        <a
          href="https://docs.litellm.ai/docs/mcp_aws_sigv4"
          target="_blank"
          rel="noopener noreferrer"
          className="text-info hover:text-info/80"
        >
          {t("mcpServers.awsSigV4Fields.viewDocs")}
        </a>
      </p>
      <MountedFormField
        label={
          <FieldLabel
            label={t("mcpServers.awsSigV4Fields.region.label")}
            tooltip={t("mcpServers.awsSigV4Fields.region.tooltip")}
          />
        }
        name={["credentials", "aws_region_name"]}
        required
        rules={{ validate: { required: requiredRule(t("mcpServers.awsSigV4Fields.region.required")) } }}
      >
        {(control) => <Input {...textControl(control)} placeholder="us-east-1" className={fieldClassName} />}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("mcpServers.awsSigV4Fields.serviceName.label")}
            tooltip={t("mcpServers.awsSigV4Fields.serviceName.tooltip")}
          />
        }
        name={["credentials", "aws_service_name"]}
      >
        {(control) => <Input {...textControl(control)} placeholder="bedrock-agentcore" className={fieldClassName} />}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("mcpServers.awsSigV4Fields.accessKeyId.label")}
            tooltip={t("mcpServers.awsSigV4Fields.accessKeyId.tooltip")}
          />
        }
        name={ACCESS_KEY_PATH}
        rules={{
          deps: ["credentials.aws_secret_access_key"],
          validate: {
            pairedWithSecret: requiredWhenSiblingSet(
              SECRET_KEY_PATH,
              t("mcpServers.awsSigV4Fields.accessKeyId.requiredWithSecret"),
            ),
          },
        }}
      >
        {(control) => (
          <PasswordInput
            {...textControl(control)}
            placeholder={t("mcpServers.awsSigV4Fields.accessKeyId.placeholder")}
            groupClassName={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("mcpServers.awsSigV4Fields.secretAccessKey.label")}
            tooltip={t("mcpServers.awsSigV4Fields.secretAccessKey.tooltip")}
          />
        }
        name={SECRET_KEY_PATH}
        rules={{
          deps: ["credentials.aws_access_key_id"],
          validate: {
            pairedWithAccessKey: requiredWhenSiblingSet(
              ACCESS_KEY_PATH,
              t("mcpServers.awsSigV4Fields.secretAccessKey.requiredWithAccessKey"),
            ),
          },
        }}
      >
        {(control) => (
          <PasswordInput
            {...textControl(control)}
            placeholder={t("mcpServers.awsSigV4Fields.secretAccessKey.placeholder")}
            groupClassName={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("mcpServers.awsSigV4Fields.sessionToken.label")}
            tooltip={t("mcpServers.awsSigV4Fields.sessionToken.tooltip")}
          />
        }
        name={["credentials", "aws_session_token"]}
      >
        {(control) => (
          <PasswordInput
            {...textControl(control)}
            placeholder={t("mcpServers.awsSigV4Fields.sessionToken.placeholder")}
            groupClassName={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("mcpServers.awsSigV4Fields.roleArn.label")}
            tooltip={t("mcpServers.awsSigV4Fields.roleArn.tooltip")}
          />
        }
        name={["credentials", "aws_role_name"]}
      >
        {(control) => (
          <Input
            {...textControl(control)}
            placeholder={t("mcpServers.awsSigV4Fields.roleArn.placeholder")}
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label={t("mcpServers.awsSigV4Fields.sessionName.label")}
            tooltip={t("mcpServers.awsSigV4Fields.sessionName.tooltip")}
          />
        }
        name={["credentials", "aws_session_name"]}
      >
        {(control) => (
          <Input
            {...textControl(control)}
            placeholder={t("mcpServers.awsSigV4Fields.sessionName.placeholder")}
            className={fieldClassName}
          />
        )}
      </MountedFormField>
    </>
  );
};

export default AwsSigV4Fields;
