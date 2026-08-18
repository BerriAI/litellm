import React from "react";
import { Form, Input, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";

const AwsSigV4Fields: React.FC = () => (
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
    <Form.Item
      label={
        <span className="text-sm font-medium text-gray-700 flex items-center">
          AWS Region
          <Tooltip title="AWS region for SigV4 signing (e.g., us-east-1)">
            <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
          </Tooltip>
        </span>
      }
      name={["credentials", "aws_region_name"]}
      rules={[{ required: true, message: "AWS region is required for SigV4 auth" }]}
    >
      <Input placeholder="us-east-1" className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500" />
    </Form.Item>
    <Form.Item
      label={
        <span className="text-sm font-medium text-gray-700 flex items-center">
          AWS Service Name
          <Tooltip title="AWS service name for SigV4 signing. Defaults to 'bedrock-agentcore'.">
            <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
          </Tooltip>
        </span>
      }
      name={["credentials", "aws_service_name"]}
    >
      <Input
        placeholder="bedrock-agentcore"
        className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
      />
    </Form.Item>
    <Form.Item
      label={
        <span className="text-sm font-medium text-gray-700 flex items-center">
          AWS Access Key ID
          <Tooltip title="Optional. If not provided, falls back to the boto3 credential chain (IAM role, env vars, etc.).">
            <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
          </Tooltip>
        </span>
      }
      name={["credentials", "aws_access_key_id"]}
      dependencies={[["credentials", "aws_secret_access_key"]]}
      rules={[
        ({ getFieldValue }) => ({
          validator(_, value) {
            const secretKey = getFieldValue(["credentials", "aws_secret_access_key"]);
            if (secretKey && !value) {
              return Promise.reject(new Error("Access Key ID is required when Secret Access Key is provided"));
            }
            return Promise.resolve();
          },
        }),
      ]}
    >
      <Input.Password
        placeholder="AKIA... (optional — uses IAM role if blank)"
        className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
      />
    </Form.Item>
    <Form.Item
      label={
        <span className="text-sm font-medium text-gray-700 flex items-center">
          AWS Secret Access Key
          <Tooltip title="Optional. Required if AWS Access Key ID is provided.">
            <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
          </Tooltip>
        </span>
      }
      name={["credentials", "aws_secret_access_key"]}
      dependencies={[["credentials", "aws_access_key_id"]]}
      rules={[
        ({ getFieldValue }) => ({
          validator(_, value) {
            const accessKeyId = getFieldValue(["credentials", "aws_access_key_id"]);
            if (accessKeyId && !value) {
              return Promise.reject(new Error("Secret Access Key is required when Access Key ID is provided"));
            }
            return Promise.resolve();
          },
        }),
      ]}
    >
      <Input.Password
        placeholder="Enter secret key (optional — uses IAM role if blank)"
        className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
      />
    </Form.Item>
    <Form.Item
      label={
        <span className="text-sm font-medium text-gray-700 flex items-center">
          AWS Session Token
          <Tooltip title="Optional. Only needed for temporary STS credentials.">
            <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
          </Tooltip>
        </span>
      }
      name={["credentials", "aws_session_token"]}
    >
      <Input.Password
        placeholder="Enter session token (optional)"
        className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
      />
    </Form.Item>
    <Form.Item
      label={
        <span className="text-sm font-medium text-gray-700 flex items-center">
          AWS Role ARN
          <Tooltip title="Optional. IAM role ARN to assume via STS before signing. If set, LiteLLM calls sts:AssumeRole to get temporary credentials. Uses ambient credentials (IAM role, env vars) as the source identity unless explicit keys are also provided.">
            <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
          </Tooltip>
        </span>
      }
      name={["credentials", "aws_role_name"]}
    >
      <Input
        placeholder="arn:aws:iam::123456789012:role/MyRole (optional)"
        className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
      />
    </Form.Item>
    <Form.Item
      label={
        <span className="text-sm font-medium text-gray-700 flex items-center">
          AWS Session Name
          <Tooltip title="Optional. Session name for the AssumeRole call — appears in CloudTrail logs. Auto-generated if omitted.">
            <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
          </Tooltip>
        </span>
      }
      name={["credentials", "aws_session_name"]}
    >
      <Input
        placeholder="litellm-prod (optional, auto-generated if blank)"
        className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
      />
    </Form.Item>
  </>
);

export default AwsSigV4Fields;
