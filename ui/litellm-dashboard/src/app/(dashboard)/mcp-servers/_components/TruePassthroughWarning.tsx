import React from "react";
import { Alert } from "antd";
import { AUTH_TYPE } from "@/components/mcp_tools/types";

/**
 * Warning shown in the create/edit MCP server forms when auth_type
 * true_passthrough is selected: the gateway performs no admission auth for
 * that server, so callers reach the upstream without a LiteLLM identity.
 */
export default function TruePassthroughWarning({ authType }: { authType?: string | null }) {
  if (authType !== AUTH_TYPE.TRUE_PASSTHROUGH) return null;
  return (
    <Alert
      type="warning"
      showIcon
      className="mb-4 rounded-lg"
      message="True Passthrough disables LiteLLM authentication for this server"
      description="Anyone who can reach the gateway can call this server without a LiteLLM key. The caller's Authorization header is forwarded to the upstream verbatim, per-key and per-team rate limits and spend tracking do not apply, and the upstream is fully responsible for authenticating callers. Choose OAuth Delegate instead if callers should still authenticate to LiteLLM."
    />
  );
}
