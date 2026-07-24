import React from "react";
import { Form, Select, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";

const TOKEN_ENDPOINT_AUTH_METHOD_OPTIONS = [
  { value: "client_secret_basic", label: "Client Secret Basic" },
  { value: "client_secret_post", label: "Client Secret Post" },
];

interface TokenEndpointAuthMethodFieldProps {
  isEditing?: boolean;
}

const TokenEndpointAuthMethodField: React.FC<TokenEndpointAuthMethodFieldProps> = ({ isEditing = false }) => (
  <Form.Item
    label={
      <span className="text-sm font-medium text-gray-700 flex items-center">
        Token Endpoint Auth Method (optional)
        <Tooltip title="How the proxy authenticates to the upstream OAuth token endpoint. Client Secret Basic sends the client credentials in an HTTP Basic Authorization header; leave blank to use the default, Client Secret Post, which sends them in the request body.">
          <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
        </Tooltip>
      </span>
    }
    name={["credentials", "token_endpoint_auth_method"]}
  >
    <Select
      allowClear
      placeholder={
        isEditing ? "Leave blank to keep existing (default Client Secret Post)" : "Default (Client Secret Post)"
      }
      className="rounded-lg"
      size="large"
      options={TOKEN_ENDPOINT_AUTH_METHOD_OPTIONS}
    />
  </Form.Item>
);

export default TokenEndpointAuthMethodField;
