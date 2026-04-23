import React from "react";
import { Form, InputNumber, Select } from "antd";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from "lucide-react";
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

const fieldClassName = "rounded-lg";

const FieldLabel: React.FC<{ label: string; tooltip: string }> = ({
  label,
  tooltip,
}) => (
  <span className="text-sm font-medium text-foreground flex items-center">
    {label}
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Info className="ml-2 h-3 w-3 text-primary cursor-help" />
        </TooltipTrigger>
        <TooltipContent className="max-w-xs">{tooltip}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
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
              <span className="text-muted-foreground text-xs ml-2">
                server-to-server, no user interaction
              </span>
            </div>
          </Select.Option>
          <Select.Option value={OAUTH_FLOW.INTERACTIVE}>
            <div>
              <span className="font-medium">Interactive (PKCE)</span>
              <span className="text-muted-foreground text-xs ml-2">
                browser-based user authorization
              </span>
            </div>
          </Select.Option>
        </Select>
      </Form.Item>

      {isM2M ? (
        <>
          <Form.Item
            label={<FieldLabel label="Client ID" tooltip="OAuth2 client ID for the client_credentials grant." />}
            name={["credentials", "client_id"]}
            rules={[
              { required: true, message: "Client ID is required for M2M OAuth" },
            ]}
          >
            <Input
              type="password"
              placeholder={`Enter OAuth client ID${placeholderSuffix}`}
              className={fieldClassName}
            />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label="Client Secret"
                tooltip="OAuth2 client secret for the client_credentials grant."
              />
            }
            name={["credentials", "client_secret"]}
            rules={[
              {
                required: true,
                message: "Client Secret is required for M2M OAuth",
              },
            ]}
          >
            <Input
              type="password"
              placeholder={`Enter OAuth client secret${placeholderSuffix}`}
              className={fieldClassName}
            />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label="Token URL"
                tooltip="Token endpoint URL for the client_credentials grant."
              />
            }
            name="token_url"
            rules={[
              { required: true, message: "Token URL is required for M2M OAuth" },
            ]}
          >
            <Input
              placeholder="https://auth.example.com/oauth/token"
              className={fieldClassName}
            />
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
                    className="text-xs text-primary hover:text-primary/80 ml-2 font-normal"
                    onClick={(e) => e.stopPropagation()}
                  >
                    Create OAuth App →
                  </a>
                )}
              </span>
            }
            name={["credentials", "client_id"]}
          >
            <Input
              type="password"
              placeholder={`Enter client ID${placeholderSuffix}`}
              className={fieldClassName}
            />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label="Client Secret (optional)"
                tooltip="Provide only if your MCP server cannot handle dynamic client registration."
              />
            }
            name={["credentials", "client_secret"]}
          >
            <Input
              type="password"
              placeholder={`Enter client secret${placeholderSuffix}`}
              className={fieldClassName}
            />
          </Form.Item>
          <Form.Item
            label={<FieldLabel label="Scopes (optional)" tooltip="Optional scopes requested during token exchange. Separate multiple scopes with enter or commas." />}
            name={["credentials", "scopes"]}
          >
            <Select mode="tags" tokenSeparators={[","]} placeholder="Add scopes" className="rounded-lg" size="large" />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label="Authorization URL (optional)"
                tooltip="Optional override for the authorization endpoint."
              />
            }
            name="authorization_url"
          >
            <Input
              placeholder="https://example.com/oauth/authorize"
              className={fieldClassName}
            />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label="Token URL (optional)"
                tooltip="Optional override for the token endpoint."
              />
            }
            name="token_url"
          >
            <Input
              placeholder="https://example.com/oauth/token"
              className={fieldClassName}
            />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label="Registration URL (optional)"
                tooltip="Optional override for the dynamic client registration endpoint."
              />
            }
            name="registration_url"
          >
            <Input
              placeholder="https://example.com/oauth/register"
              className={fieldClassName}
            />
          </Form.Item>
          <Form.Item
            label={
              <FieldLabel
                label="Token Validation Rules (optional)"
                tooltip='JSON object of key-value rules checked against the OAuth token response before storing. Supports dot-notation for nested fields (e.g. {"organization": "my-org", "team.id": "123"}). Tokens that fail validation are rejected with HTTP 403.'
              />
            }
            name="token_validation_json"
            rules={[
              {
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                validator: (_: any, value: string) => {
                  if (!value || value.trim() === "") return Promise.resolve();
                  try {
                    JSON.parse(value);
                    return Promise.resolve();
                  } catch {
                    return Promise.reject(new Error("Must be valid JSON"));
                  }
                },
              },
            ]}
          >
            <Textarea
              placeholder={
                '{\n  "organization": "my-org",\n  "team.id": "123"\n}'
              }
              rows={4}
              className="font-mono text-sm rounded-lg"
            />
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
            <div className="rounded-lg border border-dashed border-border p-4 space-y-2">
              <p className="text-sm text-muted-foreground">
                Use OAuth to fetch a fresh access token and temporarily save it
                in the session as the authentication value.
              </p>
              <Button
                type="button"
                variant="secondary"
                onClick={oauthFlow.startOAuthFlow}
                disabled={
                  oauthFlow.status === "authorizing" ||
                  oauthFlow.status === "exchanging"
                }
              >
                {oauthFlow.status === "authorizing"
                  ? "Waiting for authorization..."
                  : oauthFlow.status === "exchanging"
                    ? "Exchanging authorization code..."
                    : "Authorize & Fetch Token"}
              </Button>
              {oauthFlow.error && (
                <p className="text-sm text-destructive">{oauthFlow.error}</p>
              )}
              {oauthFlow.status === "success" &&
                oauthFlow.tokenResponse?.access_token && (
                  <p className="text-sm text-emerald-600">
                    Token fetched. Expires in{" "}
                    {oauthFlow.tokenResponse.expires_in ?? "?"} seconds.
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
