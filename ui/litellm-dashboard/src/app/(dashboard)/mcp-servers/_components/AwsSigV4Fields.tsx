import React from "react";
import { Input, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { useFormContext } from "react-hook-form";
import { MountedFormField, bindControl, type MountedFormValues } from "@/components/common_components/MountedFormField";

const fieldClassName = "rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500";

const FieldLabel: React.FC<{ label: string; tooltip: string }> = ({ label, tooltip }) => (
  <span className="text-sm font-medium text-gray-700 flex items-center">
    {label}
    <Tooltip title={tooltip}>
      <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
    </Tooltip>
  </span>
);

const AwsSigV4Fields: React.FC = () => {
  const { getValues } = useFormContext<MountedFormValues>();

  return (
    <>
      <p className="text-sm text-gray-500 mb-2">
        For MCP servers hosted on AWS Bedrock AgentCore.{" "}
        <a
          href="https://docs.litellm.ai/docs/mcp_aws_sigv4"
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-500 hover:text-blue-700"
        >
          View docs &rarr;
        </a>
      </p>
      <MountedFormField
        label={<FieldLabel label="AWS Region" tooltip="AWS region for SigV4 signing (e.g., us-east-1)" />}
        name="credentials.aws_region_name"
        required
        rules={{ required: "AWS region is required for SigV4 auth" }}
      >
        {(field) => (
          <Input {...bindControl<string | undefined>(field)} placeholder="us-east-1" className={fieldClassName} />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label="AWS Service Name"
            tooltip="AWS service name for SigV4 signing. Defaults to 'bedrock-agentcore'."
          />
        }
        name="credentials.aws_service_name"
      >
        {(field) => (
          <Input
            {...bindControl<string | undefined>(field)}
            placeholder="bedrock-agentcore"
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label="AWS Access Key ID"
            tooltip="Optional. If not provided, falls back to the boto3 credential chain (IAM role, env vars, etc.)."
          />
        }
        name="credentials.aws_access_key_id"
        rules={{
          deps: ["credentials.aws_secret_access_key"],
          validate: (value) =>
            getValues("credentials.aws_secret_access_key") && !value
              ? "Access Key ID is required when Secret Access Key is provided"
              : true,
        }}
      >
        {(field) => (
          <Input.Password
            {...bindControl<string | undefined>(field)}
            placeholder="AKIA... (optional — uses IAM role if blank)"
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel label="AWS Secret Access Key" tooltip="Optional. Required if AWS Access Key ID is provided." />
        }
        name="credentials.aws_secret_access_key"
        rules={{
          deps: ["credentials.aws_access_key_id"],
          validate: (value) =>
            getValues("credentials.aws_access_key_id") && !value
              ? "Secret Access Key is required when Access Key ID is provided"
              : true,
        }}
      >
        {(field) => (
          <Input.Password
            {...bindControl<string | undefined>(field)}
            placeholder="Enter secret key (optional — uses IAM role if blank)"
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={<FieldLabel label="AWS Session Token" tooltip="Optional. Only needed for temporary STS credentials." />}
        name="credentials.aws_session_token"
      >
        {(field) => (
          <Input.Password
            {...bindControl<string | undefined>(field)}
            placeholder="Enter session token (optional)"
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label="AWS Role ARN"
            tooltip="Optional. IAM role ARN to assume via STS before signing. If set, LiteLLM calls sts:AssumeRole to get temporary credentials. Uses ambient credentials (IAM role, env vars) as the source identity unless explicit keys are also provided."
          />
        }
        name="credentials.aws_role_name"
      >
        {(field) => (
          <Input
            {...bindControl<string | undefined>(field)}
            placeholder="arn:aws:iam::123456789012:role/MyRole (optional)"
            className={fieldClassName}
          />
        )}
      </MountedFormField>
      <MountedFormField
        label={
          <FieldLabel
            label="AWS Session Name"
            tooltip="Optional. Session name for the AssumeRole call — appears in CloudTrail logs. Auto-generated if omitted."
          />
        }
        name="credentials.aws_session_name"
      >
        {(field) => (
          <Input
            {...bindControl<string | undefined>(field)}
            placeholder="litellm-prod (optional, auto-generated if blank)"
            className={fieldClassName}
          />
        )}
      </MountedFormField>
    </>
  );
};

export default AwsSigV4Fields;
