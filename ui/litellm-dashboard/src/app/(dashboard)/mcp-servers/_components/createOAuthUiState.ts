import { MCPServerCostInfo, withoutMintedTokenCredentials } from "@/components/mcp_tools/types";
import { getSecureItem, setSecureItem } from "@/utils/secureStorage";

const CREATE_OAUTH_UI_STATE_KEY = "litellm-mcp-oauth-create-state";

// Everything the create modal needs to look untouched after the OAuth authorize redirect reloads the
// page. `authorizedIdentity` is part of it so invalidation stays armed across the round trip: without
// it the remounted form starts with no identity, and a post-restore url/mode edit would never fire the
// stale-token discard.
export interface CreateUiSnapshot {
  readonly modalVisible: boolean;
  readonly formValues: Record<string, unknown>;
  readonly transportType: string;
  readonly costConfig: MCPServerCostInfo;
  readonly allowedTools: readonly string[];
  readonly hasToolAllowlistInteraction: boolean;
  readonly aliasManuallyEdited: boolean;
  readonly logoUrl: string | undefined;
  readonly authorizedIdentity: string | undefined;
}

// Only the fields that survived their own presence check. A key absent here means "leave the freshly
// mounted state alone", which is why every field is optional rather than defaulted.
export type RestoredUiSnapshot = {
  readonly modalVisible?: boolean;
  readonly formValues?: Record<string, unknown>;
  readonly transportType?: string;
  readonly costConfig?: MCPServerCostInfo;
  readonly allowedTools?: readonly string[];
  readonly hasToolAllowlistInteraction?: boolean;
  readonly aliasManuallyEdited?: boolean;
  readonly logoUrl?: string;
  readonly authorizedIdentity?: string;
};

export const writeCreateUiSnapshot = (snapshot: CreateUiSnapshot): void => {
  if (typeof window === "undefined") {
    return;
  }
  try {
    setSecureItem(CREATE_OAUTH_UI_STATE_KEY, JSON.stringify(snapshot));
  } catch (err) {
    console.warn("Failed to persist MCP create state", err);
  }
};

/**
 * Read and validate the snapshot left before the authorize redirect, then drop it so a later mount
 * cannot replay it. Returns null when there is nothing to restore (or the payload was unparseable),
 * in which case the stored value is left in place for an in-flight flow to time out naturally.
 */
export const readCreateUiSnapshot = (): RestoredUiSnapshot | null => {
  if (typeof window === "undefined") {
    return null;
  }
  const storedState = getSecureItem(CREATE_OAUTH_UI_STATE_KEY);
  if (!storedState) {
    return null;
  }

  try {
    const parsed = JSON.parse(storedState);
    const restoredTransport = parsed.formValues?.transport || parsed.transportType || "";

    return {
      ...(parsed.modalVisible ? { modalVisible: true } : {}),
      ...(restoredTransport ? { transportType: restoredTransport } : {}),
      ...(parsed.formValues
        ? {
            // Strip minted token material so a stale token never rehydrates; the declared app the
            // admin typed is kept. Create has no server-side stored app to merge.
            formValues: {
              ...parsed.formValues,
              credentials: withoutMintedTokenCredentials(parsed.formValues.credentials),
            },
          }
        : {}),
      ...(typeof parsed.authorizedIdentity === "string" ? { authorizedIdentity: parsed.authorizedIdentity } : {}),
      ...(parsed.costConfig ? { costConfig: parsed.costConfig } : {}),
      ...(parsed.allowedTools ? { allowedTools: parsed.allowedTools } : {}),
      ...(typeof parsed.hasToolAllowlistInteraction === "boolean"
        ? { hasToolAllowlistInteraction: parsed.hasToolAllowlistInteraction }
        : {}),
      ...(typeof parsed.aliasManuallyEdited === "boolean" ? { aliasManuallyEdited: parsed.aliasManuallyEdited } : {}),
      ...(parsed.logoUrl ? { logoUrl: parsed.logoUrl } : {}),
    };
  } catch (err) {
    console.error("Failed to restore MCP create state", err);
    return null;
  } finally {
    window.sessionStorage.removeItem(CREATE_OAUTH_UI_STATE_KEY);
  }
};
