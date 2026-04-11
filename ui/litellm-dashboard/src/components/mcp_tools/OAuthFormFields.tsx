import React, { useState } from "react";
import { Form, Input, InputNumber, Select, Tooltip } from "antd";
import { InfoCircleOutlined, PlusOutlined, CloseOutlined } from "@ant-design/icons";
import { Button, TextInput } from "@tremor/react";
import { OAUTH_FLOW } from "./types";

const CLAIM_PRESETS = [
  { label: "Slack enterprise", field: "enterprise_id", placeholder: "e.g. E04XXXXXXX" },
  { label: "Jira cloud", field: "cloud_id", placeholder: "e.g. abc-123" },
  { label: "GitHub Enterprise", field: "enterprise", placeholder: "e.g. my-org" },
];

interface ClaimRow { field: string; value: string }

interface TokenClaimsBuilderProps {
  value?: string; // JSON string from antd Form
  onChange?: (v: string) => void;
}

const TokenClaimsBuilder: React.FC<TokenClaimsBuilderProps> = ({ value, onChange }) => {
  const parse = (v?: string): ClaimRow[] => {
    if (!v || v.trim() === "") return [];
    try {
      const obj = JSON.parse(v);
      return Object.entries(obj).map(([field, val]) => ({ field, value: String(val) }));
    } catch {
      return [];
    }
  };

  const [rows, setRows] = useState<ClaimRow[]>(() => parse(value));

  const emit = (next: ClaimRow[]) => {
    setRows(next);
    const nonempty = next.filter(r => r.field.trim() !== "");
    if (nonempty.length === 0) { onChange?.(""); return; }
    const obj: Record<string, string> = {};
    nonempty.forEach(r => { obj[r.field.trim()] = r.value; });
    onChange?.(JSON.stringify(obj));
  };

  const update = (i: number, key: keyof ClaimRow, val: string) =>
    emit(rows.map((r, idx) => idx === i ? { ...r, [key]: val } : r));

  const remove = (i: number) => emit(rows.filter((_, idx) => idx !== i));

  const addPreset = (field: string) => {
    if (rows.some(r => r.field === field)) return;
    emit([...rows, { field, value: "" }]);
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        <span className="text-xs text-gray-400 self-center">Quick add:</span>
        {CLAIM_PRESETS.map(p => (
          <button
            key={p.field}
            type="button"
            onClick={() => addPreset(p.field)}
            disabled={rows.some(r => r.field === p.field)}
            className="px-2 py-0.5 text-xs rounded-full border border-blue-300 text-blue-600 hover:bg-blue-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {p.label}
          </button>
        ))}
      </div>

      {rows.length > 0 && (
        <div className="rounded-lg border border-gray-200 overflow-hidden">
          <div className="grid grid-cols-[1fr_1fr_32px] bg-gray-50 border-b border-gray-200 px-3 py-1.5">
            <span className="text-xs font-medium text-gray-500">Token field</span>
            <span className="text-xs font-medium text-gray-500">Required value</span>
            <span />
          </div>
          {rows.map((row, i) => {
            const preset = CLAIM_PRESETS.find(p => p.field === row.field);
            return (
              <div key={i} className="grid grid-cols-[1fr_1fr_32px] items-center px-3 py-2 gap-2 border-b border-gray-100 last:border-b-0 hover:bg-gray-50">
                <input
                  className="w-full text-sm border border-gray-200 rounded px-2 py-1 focus:outline-none focus:border-blue-400"
                  placeholder="e.g. enterprise_id"
                  value={row.field}
                  onChange={e => update(i, "field", e.target.value)}
                />
                <input
                  className="w-full text-sm border border-gray-200 rounded px-2 py-1 focus:outline-none focus:border-blue-400"
                  placeholder={preset?.placeholder ?? "required value"}
                  value={row.value}
                  onChange={e => update(i, "value", e.target.value)}
                />
                <button
                  type="button"
                  onClick={() => remove(i)}
                  className="flex items-center justify-center w-6 h-6 rounded hover:bg-red-50 text-gray-400 hover:text-red-500 transition-colors"
                >
                  <CloseOutlined style={{ fontSize: 11 }} />
                </button>
              </div>
            );
          })}
        </div>
      )}

      <button
        type="button"
        onClick={() => emit([...rows, { field: "", value: "" }])}
        className="flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 transition-colors"
      >
        <PlusOutlined style={{ fontSize: 11 }} />
        Add claim
      </button>
    </div>
  );
};

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
  const placeholderSuffix = isEditing ? " (leave blank to keep existing)" : "";

  return (
    <>
      <Form.Item
        label={
          <FieldLabel
            label="OAuth Flow Type"
            tooltip="Choose how the proxy authenticates with this MCP server. M2M is for server-to-server communication using client credentials. Interactive (PKCE) is for user-facing flows that require browser-based authorization."
          />
        }
        name="oauth_flow_type"
        {...(initialFlowType ? { initialValue: initialFlowType } : {})}
      >
        <Select className="rounded-lg" size="large">
          <Select.Option value={OAUTH_FLOW.M2M}>
            <div>
              <span className="font-medium">Machine-to-Machine (M2M)</span>
              <span className="text-gray-400 text-xs ml-2">server-to-server, no user interaction</span>
            </div>
          </Select.Option>
          <Select.Option value={OAUTH_FLOW.INTERACTIVE}>
            <div>
              <span className="font-medium">Interactive (PKCE)</span>
              <span className="text-gray-400 text-xs ml-2">browser-based user authorization</span>
            </div>
          </Select.Option>
        </Select>
      </Form.Item>

      {isM2M ? (
        <>
          <Form.Item
            label={<FieldLabel label="Client ID" tooltip="OAuth2 client ID for the client_credentials grant." />}
            name={["credentials", "client_id"]}
            rules={[{ required: true, message: "Client ID is required for M2M OAuth" }]}
          >
            <TextInput type="password" placeholder={`Enter OAuth client ID${placeholderSuffix}`} className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={<FieldLabel label="Client Secret" tooltip="OAuth2 client secret for the client_credentials grant." />}
            name={["credentials", "client_secret"]}
            rules={[{ required: true, message: "Client Secret is required for M2M OAuth" }]}
          >
            <TextInput type="password" placeholder={`Enter OAuth client secret${placeholderSuffix}`} className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={<FieldLabel label="Token URL" tooltip="Token endpoint URL for the client_credentials grant." />}
            name="token_url"
            rules={[{ required: true, message: "Token URL is required for M2M OAuth" }]}
          >
            <TextInput placeholder="https://auth.example.com/oauth/token" className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={<FieldLabel label="Scopes (optional)" tooltip="Optional scopes to request with the client_credentials grant." />}
            name={["credentials", "scopes"]}
          >
            <Select mode="tags" tokenSeparators={[","]} placeholder="Add scopes" className="rounded-lg" size="large" />
          </Form.Item>
        </>
      ) : (
        <>
          <Form.Item
            label={
              <span className="flex items-center justify-between w-full">
                <FieldLabel label="Client ID (optional)" tooltip="Provide only if your MCP server cannot handle dynamic client registration." />
                {docsUrl && (
                  <a
                    href={docsUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-xs text-blue-500 hover:text-blue-700 ml-2 font-normal"
                    onClick={(e) => e.stopPropagation()}
                  >
                    Create OAuth App →
                  </a>
                )}
              </span>
            }
            name={["credentials", "client_id"]}
          >
            <TextInput type="password" placeholder={`Enter client ID${placeholderSuffix}`} className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={<FieldLabel label="Client Secret (optional)" tooltip="Provide only if your MCP server cannot handle dynamic client registration." />}
            name={["credentials", "client_secret"]}
          >
            <TextInput type="password" placeholder={`Enter client secret${placeholderSuffix}`} className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={<FieldLabel label="Scopes (optional)" tooltip="Optional scopes requested during token exchange. Separate multiple scopes with enter or commas." />}
            name={["credentials", "scopes"]}
          >
            <Select mode="tags" tokenSeparators={[","]} placeholder="Add scopes" className="rounded-lg" size="large" />
          </Form.Item>
          <Form.Item
            label={<FieldLabel label="Authorization URL (optional)" tooltip="Optional override for the authorization endpoint." />}
            name="authorization_url"
          >
            <TextInput placeholder="https://example.com/oauth/authorize" className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={<FieldLabel label="Token URL (optional)" tooltip="Optional override for the token endpoint." />}
            name="token_url"
          >
            <TextInput placeholder="https://example.com/oauth/token" className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={<FieldLabel label="Registration URL (optional)" tooltip="Optional override for the dynamic client registration endpoint." />}
            name="registration_url"
          >
            <TextInput placeholder="https://example.com/oauth/register" className={fieldClassName} />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label="Required Token Claims (optional)"
                tooltip="Block personal accounts by requiring specific fields in the OAuth token response. For example, require enterprise_id to match your Slack workspace. Tokens that don't match are rejected with HTTP 403."
              />
            }
            name="token_validation_json"
          >
            <TokenClaimsBuilder />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label="Token Storage TTL (seconds, optional)"
                tooltip="How long to cache each user's OAuth access token in Redis before evicting it (regardless of the token's own expires_in). Leave blank to derive the TTL from the token's expires_in, or fall back to the 12-hour default."
              />
            }
            name="token_storage_ttl_seconds"
          >
            <InputNumber
              min={1}
              placeholder="e.g. 3600"
              className="w-full rounded-lg"
              style={{ width: "100%" }}
            />
          </Form.Item>
          {oauthFlow && (
            <div className="rounded-lg border border-dashed border-gray-300 p-4 space-y-2">
              <p className="text-sm text-gray-600">
                Use OAuth to fetch a fresh access token and temporarily save it in the session as the authentication value.
              </p>
              <Button
                variant="secondary"
                onClick={oauthFlow.startOAuthFlow}
                disabled={oauthFlow.status === "authorizing" || oauthFlow.status === "exchanging"}
              >
                {oauthFlow.status === "authorizing"
                  ? "Waiting for authorization..."
                  : oauthFlow.status === "exchanging"
                    ? "Exchanging authorization code..."
                    : "Authorize & Fetch Token"}
              </Button>
              {oauthFlow.error && <p className="text-sm text-red-500">{oauthFlow.error}</p>}
              {oauthFlow.status === "success" && oauthFlow.tokenResponse?.access_token && (
                <p className="text-sm text-green-600">
                  Token fetched. Expires in {oauthFlow.tokenResponse.expires_in ?? "?"} seconds.
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
