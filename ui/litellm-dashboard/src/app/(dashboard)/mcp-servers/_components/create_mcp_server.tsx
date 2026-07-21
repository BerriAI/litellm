import React, { useState } from "react";
import { Modal, Tooltip, Form, Select, Input, InputNumber, Switch, Collapse } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Button, TextInput } from "@tremor/react";
import { createMCPServer, registerMCPServer, storeMCPOAuthUserCredential } from "@/components/networking";
import { setToken } from "@/utils/mcpTokenStore";
import {
  AUTH_TYPE,
  DiscoverableMCPServer,
  OAUTH_FLOW,
  MCPServer,
  MCPServerCostInfo,
  TRANSPORT,
  getMcpOAuthMode,
  MCP_OAUTH2_FLOW_M2M,
  MCP_OAUTH2_FLOW_INTERACTIVE,
  isClientForwardedTokenMode,
  getOAuthAuthorizationIdentity,
  CLEARED_ON_INVALIDATION,
  isHeldOAuthTokenStale,
  preservedDeclaredAppCredentials,
  withoutMintedTokenCredentials,
} from "@/components/mcp_tools/types";
import OAuthFormFields from "./OAuthFormFields";
import TruePassthroughWarning from "./TruePassthroughWarning";
import PassthroughAuthorizeSection from "./PassthroughAuthorizeSection";
import TokenExchangeFormFields from "./TokenExchangeFormFields";
import IdJagFormFields from "./IdJagFormFields";
import MCPServerCostConfig from "./mcp_server_cost_config";
import MCPConnectionStatus from "./mcp_connection_status";
import MCPToolConfiguration from "./mcp_tool_configuration";
import StdioConfiguration from "./StdioConfiguration";
import MCPPermissionManagement from "./MCPPermissionManagement";
import OpenAPIFormSection, { OpenAPIKeyTool } from "./OpenAPIFormSection";
import MCPLogoSelector from "./MCPLogoSelector";
import EnvVarsSection from "./EnvVarsSection";
import { isAdminRole } from "@/utils/roles";
import { validateMCPServerUrl, validateMCPServerName, normalizeEnvVars, TOOL_DISPLAY_NAME_PATTERN } from "./utils";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { useMcpOAuthFlow } from "@/hooks/useMcpOAuthFlow";
import { useTestMCPConnection } from "@/hooks/useTestMCPConnection";
import { getSecureItem, setSecureItem } from "@/utils/secureStorage";
import { resolveLogoSrc } from "@/lib/assetPaths";

const asset_logos_folder = "/ui/assets/logos/";
export const mcpLogoImg = `${asset_logos_folder}mcp_logo.png`;

interface CreateMCPServerProps {
  userRole: string;
  userID?: string | null;
  accessToken: string | null;
  onCreateSuccess: (newMcpServer: MCPServer) => void;
  isModalVisible: boolean;
  setModalVisible: (visible: boolean) => void;
  availableAccessGroups: string[];
  prefillData?: DiscoverableMCPServer | null;
  onBackToDiscovery?: () => void;
}

const AUTH_TYPES_REQUIRING_AUTH_VALUE = [AUTH_TYPE.API_KEY, AUTH_TYPE.BEARER_TOKEN, AUTH_TYPE.TOKEN, AUTH_TYPE.BASIC];
const AUTH_TYPES_REQUIRING_CREDENTIALS = [
  ...AUTH_TYPES_REQUIRING_AUTH_VALUE,
  AUTH_TYPE.OAUTH2,
  AUTH_TYPE.OAUTH2_TOKEN_EXCHANGE,
  AUTH_TYPE.OAUTH2_ID_JAG,
  AUTH_TYPE.AWS_SIGV4,
  AUTH_TYPE.TRUE_PASSTHROUGH,
  AUTH_TYPE.OAUTH_DELEGATE,
];
const CREATE_OAUTH_UI_STATE_KEY = "litellm-mcp-oauth-create-state";

const reduceStaticHeaders = (list: unknown): Record<string, string> => {
  if (!Array.isArray(list)) return {};
  return list.reduce((acc: Record<string, string>, entry: Record<string, string>) => {
    const header = entry?.header?.trim();
    if (header) acc[header] = (entry?.value ?? "").trim();
    return acc;
  }, {});
};

const CreateMCPServer: React.FC<CreateMCPServerProps> = ({
  userID,
  userRole,
  accessToken,
  onCreateSuccess,
  isModalVisible,
  setModalVisible,
  availableAccessGroups,
  prefillData,
  onBackToDiscovery,
}) => {
  const [form] = Form.useForm();
  const [isLoading, setIsLoading] = useState(false);
  const [costConfig, setCostConfig] = useState<MCPServerCostInfo>({});
  const [formValues, setFormValues] = useState<Record<string, any>>({});
  const [pendingRestoredValues, setPendingRestoredValues] = useState<{
    values: Record<string, any>;
    transport?: string;
  } | null>(null);
  const [aliasManuallyEdited, setAliasManuallyEdited] = useState(false);
  const [allowedTools, setAllowedTools] = useState<string[]>([]);
  const [hasToolAllowlistInteraction, setHasToolAllowlistInteraction] = useState(false);
  const [toolNameToDisplayName, setToolNameToDisplayName] = useState<Record<string, string>>({});
  const [toolNameToDescription, setToolNameToDescription] = useState<Record<string, string>>({});
  const [transportType, setTransportType] = useState<string>("");
  const [keyTools, setKeyTools] = useState<OpenAPIKeyTool[]>([]);
  const [searchValue, setSearchValue] = useState<string>("");
  const [oauthAccessToken, setOauthAccessToken] = useState<string | null>(null);
  const [logoUrl, setLogoUrl] = useState<string | undefined>(undefined);
  const [oauthDocsUrl, setOauthDocsUrl] = useState<string | null>(null);
  // The OAuth authorization identity (see getOAuthAuthorizationIdentity) captured at the moment a token
  // was fetched; undefined when no valid token is held. If any mint-relevant field diverges from this,
  // the held token is stale and is discarded so the admin must re-authorize.
  const [authorizedIdentity, setAuthorizedIdentity] = useState<string | undefined>(undefined);
  // The DCR-minted OAuth client from an interactive (oauth2) Authorize. Held OUT of form.credentials so
  // it can never be collected as a client-forwarded server's declared app; injected into the payload
  // only on an oauth2 submit (where persisting the registered client is correct), and cleared on any
  // invalidation or modal close. An abandoned authorize leaves it null, which is the desired asymmetry.
  const dcrClientRef = React.useRef<{ client_id: string; client_secret?: string } | null>(null);
  // Set when the upstream identity (url/endpoints) changed while a declared app is present, so the
  // section can warn that the saved app may not match the new upstream (the app is kept, not wiped).
  const [appMayNotMatchUpstream, setAppMayNotMatchUpstream] = useState(false);

  // Single hook call shared by MCPConnectionStatus and MCPToolConfiguration to avoid duplicate requests.
  const {
    tools,
    isLoadingTools,
    toolsError,
    toolsErrorStatus,
    toolsErrorStackTrace,
    canFetchTools,
    fetchTools,
    clearTools,
  } = useTestMCPConnection({
    accessToken,
    oauthAccessToken,
    formValues,
    enabled: true,
  });

  const authType = formValues.auth_type as string | undefined;
  const shouldShowAuthValueField = authType ? AUTH_TYPES_REQUIRING_AUTH_VALUE.includes(authType) : false;
  const isOAuthAuthType = authType === AUTH_TYPE.OAUTH2;
  const isTokenExchangeAuthType = authType === AUTH_TYPE.OAUTH2_TOKEN_EXCHANGE;
  const isIdJagAuthType = authType === AUTH_TYPE.OAUTH2_ID_JAG;
  const isAwsSigV4AuthType = authType === AUTH_TYPE.AWS_SIGV4;
  const isM2MFlow = isOAuthAuthType && formValues.oauth_flow_type === OAUTH_FLOW.M2M;

  const persistCreateUiState = () => {
    if (typeof window === "undefined") {
      return;
    }
    try {
      const values = form.getFieldsValue(true);
      const uiState = {
        modalVisible: isModalVisible,
        formValues: values,
        transportType,
        costConfig,
        allowedTools,
        hasToolAllowlistInteraction,
        searchValue,
        aliasManuallyEdited,
        logoUrl,
        // Persist the identity so invalidation stays armed across the OAuth redirect round trip: a
        // post-restore url/mode edit must still discard the held token instead of silently keeping it.
        authorizedIdentity,
      };
      setSecureItem(CREATE_OAUTH_UI_STATE_KEY, JSON.stringify(uiState));
    } catch (err) {
      console.warn("Failed to persist MCP create state", err);
    }
  };

  const {
    startOAuthFlow,
    status: oauthStatus,
    error: oauthError,
    tokenResponse: oauthTokenResponse,
    reset: resetOAuthFlow,
  } = useMcpOAuthFlow({
    accessToken,
    // Merge the ref-held DCR client so a re-authorize reuses the registered client instead of
    // re-registering; the form store itself never holds the DCR client (see onTokenReceived).
    getCredentials: () => ({
      ...((form.getFieldValue("credentials") as Record<string, unknown> | undefined) ?? {}),
      ...(dcrClientRef.current ?? {}),
    }),
    getTemporaryPayload: () => {
      const values = form.getFieldsValue(true);
      const transport = values.transport || transportType;
      // For OpenAPI transport the form has spec_path instead of url.
      // We pass the spec_path as url so the temp-session endpoint has something
      // to store; the backend uses authorization_url / token_url for the actual
      // OAuth redirect, so the spec_path value is never used for OAuth itself.
      const url = values.url || (transport === TRANSPORT.OPENAPI ? values.spec_path : undefined);
      if (!url || !transport) {
        return null;
      }
      const staticHeaders = reduceStaticHeaders(values.static_headers);

      return {
        server_id: undefined,
        server_name: values.server_name,
        alias: values.alias,
        description: values.description,
        url,
        transport: transport === TRANSPORT.OPENAPI ? "http" : transport,
        auth_type: isClientForwardedTokenMode(values.auth_type) ? values.auth_type : AUTH_TYPE.OAUTH2,
        // Mirror getCredentials: merge the ref-held DCR client for oauth2 so a re-authorize reuses the
        // registered client (useMcpOAuthFlow keys reuse off credentials.client_id) instead of re-DCRing;
        // the client-forwarded modes carry only the declared app.
        credentials: isClientForwardedTokenMode(values.auth_type)
          ? preservedDeclaredAppCredentials(values.credentials)
          : { ...((values.credentials as Record<string, unknown> | undefined) ?? {}), ...(dcrClientRef.current ?? {}) },
        issuer: values.issuer,
        authorization_url: values.authorization_url,
        token_url: values.token_url,
        registration_url: values.registration_url,
        mcp_access_groups: values.mcp_access_groups,
        static_headers: staticHeaders,
        command: values.command,
        args: values.args,
        env: values.env,
      };
    },
    onTokenReceived: (token, registeredClient) => {
      setOauthAccessToken(token?.access_token ?? null);

      if (!token?.access_token) {
        return;
      }

      if (isClientForwardedTokenMode(form.getFieldValue("auth_type"))) {
        // Browser-only modes: the token is held in local state (oauthAccessToken) for tool preview
        // and committed to sessionStorage on submit; it must never be written into form.credentials,
        // which would persist it as server-level credentials on the created server row. Mirrors the
        // edit form's onTokenReceived early return.
        setAuthorizedIdentity(getOAuthAuthorizationIdentity(form.getFieldsValue(true)));
        NotificationsManager.success(
          "Token held for this browser session. Tools can now be previewed and configured; the token is not saved to LiteLLM.",
        );
        return;
      }

      // The DCR-minted client is held in a ref, NOT written into form.credentials, so it can never be
      // collected as a client-forwarded server's declared app; it is injected into the payload only on
      // an oauth2 submit. An admin-typed client already lives in form.credentials and is left untouched.
      dcrClientRef.current = registeredClient?.clientId
        ? {
            client_id: registeredClient.clientId,
            ...(registeredClient.clientSecret && { client_secret: registeredClient.clientSecret }),
          }
        : null;

      const current = (form.getFieldValue("credentials") as Record<string, unknown> | undefined) ?? {};
      const nextCredentials = {
        ...(preservedDeclaredAppCredentials(current) ?? {}),
        ...(current.scopes !== undefined && { scopes: current.scopes }),
        access_token: token.access_token,
        ...(token.refresh_token && { refresh_token: token.refresh_token }),
        ...(token.expires_in && { expires_in: token.expires_in }),
        ...(token.scope && { scope: token.scope }),
      };
      // Path-replace (not deep-merge) so a re-authorize with fewer token fields does not leave stale
      // siblings from the previous token behind; the admin-typed client keys and scopes are carried
      // explicitly above.
      form.setFieldValue("credentials", nextCredentials);
      // Capture the identity AFTER writing the token so the held token is not spuriously invalidated by
      // its own credential write.
      setAuthorizedIdentity(getOAuthAuthorizationIdentity(form.getFieldsValue(true)));

      NotificationsManager.success(
        "OAuth authorization successful! Please click 'Create MCP Server' to save the configuration.",
      );
    },
    onBeforeRedirect: persistCreateUiState,
    flowSource: "create",
  });

  // Discard the held browser-authorized token and its tool preview when the authorization identity
  // changes (or the modal closes). The CLEARED_ON_INVALIDATION form fields (shared with the edit form
  // via types.tsx) are reset too; whatever the admin just changed (passed via changedValues) is
  // re-applied so the invalidation never wipes their in-flight edit. Admin-typed endpoint fields are
  // left alone (see CLEARED_ON_INVALIDATION).
  const clearHeldOAuthToken = (changedValues: Record<string, unknown> = {}) => {
    setOauthAccessToken(null);
    clearTools();
    resetOAuthFlow();
    setAuthorizedIdentity(undefined);
    dcrClientRef.current = null;
    // Capture the admin-typed app before resetFields destroys it, then re-apply it: the app is
    // upstream-scoped config, not minted material, so it survives every invalidation (the token is
    // what gets discarded). Token-shaped keys are excluded by the helper's key filter.
    const keptAppCredentials = preservedDeclaredAppCredentials(form.getFieldValue("credentials"));
    form.resetFields([...CLEARED_ON_INVALIDATION]);
    if (keptAppCredentials) {
      form.setFieldsValue({ credentials: keptAppCredentials });
    }
    // Re-apply the in-flight edit last; rc-field-form deep-merges nested objects, so a changed
    // credentials sub-field composes with the preserved sibling instead of replacing the object.
    const preserved = Object.fromEntries(
      CLEARED_ON_INVALIDATION.filter((key) => key in changedValues).map((key) => [key, changedValues[key]]),
    );
    if (Object.keys(preserved).length > 0) {
      form.setFieldsValue(preserved);
    }
  };

  React.useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const storedState = getSecureItem(CREATE_OAUTH_UI_STATE_KEY);
    if (!storedState) {
      return;
    }

    try {
      const parsed = JSON.parse(storedState);
      if (parsed.modalVisible) {
        setModalVisible(true);
      }
      const restoredTransport = parsed.formValues?.transport || parsed.transportType || "";
      if (restoredTransport) {
        setTransportType(restoredTransport);
      }
      if (parsed.formValues) {
        // Assign the cleaned credentials (strip minted token material so a stale token never rehydrates);
        // the declared app the admin typed is kept. Create has no server-side stored app to merge.
        const restoredValues = {
          ...parsed.formValues,
          credentials: withoutMintedTokenCredentials(parsed.formValues.credentials),
        };
        setPendingRestoredValues({ values: restoredValues, transport: restoredTransport });
      }
      if (typeof parsed.authorizedIdentity === "string") {
        // Re-arm invalidation: without this the remounted form has authorizedIdentity=undefined, so a
        // post-restore mode/url edit would never fire the stale-token discard.
        setAuthorizedIdentity(parsed.authorizedIdentity);
      }
      if (parsed.costConfig) {
        setCostConfig(parsed.costConfig);
      }
      if (parsed.allowedTools) {
        setAllowedTools(parsed.allowedTools);
      }
      if (typeof parsed.hasToolAllowlistInteraction === "boolean") {
        setHasToolAllowlistInteraction(parsed.hasToolAllowlistInteraction);
      }
      if (parsed.searchValue) {
        setSearchValue(parsed.searchValue);
      }
      if (typeof parsed.aliasManuallyEdited === "boolean") {
        setAliasManuallyEdited(parsed.aliasManuallyEdited);
      }
      if (parsed.logoUrl) {
        setLogoUrl(parsed.logoUrl);
      }
    } catch (err) {
      console.error("Failed to restore MCP create state", err);
    } finally {
      window.sessionStorage.removeItem(CREATE_OAUTH_UI_STATE_KEY);
    }
  }, [form, setModalVisible]);

  React.useEffect(() => {
    if (!pendingRestoredValues) {
      return;
    }
    const transportReady = transportType || pendingRestoredValues.transport || "";
    if (pendingRestoredValues.transport && !transportType) {
      // wait until transportType state catches up so the URL field is mounted
      return;
    }
    form.setFieldsValue(pendingRestoredValues.values);
    setFormValues(pendingRestoredValues.values);
    setPendingRestoredValues(null);
  }, [pendingRestoredValues, form, transportType]);

  // Pre-fill form from discovery selection
  React.useEffect(() => {
    if (!isModalVisible || !prefillData) {
      return;
    }
    // Sanitize server name: strip vendor prefix, replace hyphens with underscores
    const sanitizedName = (prefillData.name || "")
      .replace(/[^a-zA-Z0-9_]/g, "_")
      .replace(/_+/g, "_")
      .replace(/^_|_$/g, "");

    const transport = prefillData.transport || "";
    setTransportType(transport);

    const prefillValues: Record<string, any> = {
      server_name: sanitizedName,
      alias: sanitizedName,
      description: prefillData.description || "",
      transport: transport,
    };

    if (transport === "stdio") {
      const stdioObj: Record<string, any> = {};
      if (prefillData.command) stdioObj.command = prefillData.command;
      if (prefillData.args && prefillData.args.length > 0) stdioObj.args = prefillData.args;
      if (prefillData.env_vars && prefillData.env_vars.length > 0) {
        const envObj: Record<string, string> = {};
        for (const v of prefillData.env_vars) {
          envObj[v.name] = v.description ? `<${v.description}>` : "";
        }
        stdioObj.env = envObj;
      }
      if (Object.keys(stdioObj).length > 0) {
        prefillValues.stdio_config = JSON.stringify(stdioObj, null, 2);
      }
    } else if (prefillData.url) {
      prefillValues.url = prefillData.url;
    }

    form.setFieldsValue(prefillValues);
    setFormValues(prefillValues);
    setAliasManuallyEdited(false);
  }, [isModalVisible, prefillData, form]);

  const handleCreate = async (values: Record<string, any>) => {
    const invalidDisplayName = Object.entries(toolNameToDisplayName).find(
      ([, displayName]) => displayName && !TOOL_DISPLAY_NAME_PATTERN.test(displayName),
    );
    if (invalidDisplayName) {
      NotificationsManager.fromBackend(
        `Tool display name "${invalidDisplayName[1]}" is invalid. Only letters, digits, underscores, and hyphens are allowed (no spaces).`,
      );
      return;
    }
    setIsLoading(true);
    try {
      const {
        static_headers: staticHeadersList,
        env_vars: envVarsList,
        stdio_config: rawStdioConfig,
        credentials: credentialValues,
        allow_all_keys: allowAllKeysRaw,
        available_on_public_internet: availableOnPublicInternetRaw,
        delegate_auth_to_upstream: delegateAuthToUpstreamRaw,
        oauth_passthrough: oauthPassthroughRaw,
        dcr_bridge: dcrBridgeRaw,
        token_validation_json: rawTokenValidationJson,
        id_jag_client_auth_method: idJagClientAuthMethodRaw,
        ...restValues
      } = values;

      // Transform access groups into objects with name property
      const accessGroups = restValues.mcp_access_groups;

      const staticHeaders = reduceStaticHeaders(staticHeadersList);
      const envVars = normalizeEnvVars(envVarsList);

      const credentialsPayload =
        credentialValues && typeof credentialValues === "object"
          ? Object.entries(credentialValues).reduce((acc: Record<string, any>, [key, value]) => {
              if (value === undefined || value === null || value === "") {
                return acc;
              }
              if (key === "scopes") {
                if (Array.isArray(value)) {
                  const filteredScopes = value.filter((scope) => scope != null && scope !== "");
                  if (filteredScopes.length > 0) {
                    acc[key] = filteredScopes;
                  }
                }
              } else {
                acc[key] = value;
              }
              return acc;
            }, {})
          : undefined;

      // Process stdio configuration if present
      let stdioFields = {};
      if (rawStdioConfig && transportType === "stdio") {
        try {
          const stdioConfig = JSON.parse(rawStdioConfig);

          // Handle both formats:
          // 1. Full mcpServers structure: {"mcpServers": {"server-name": {...}}}
          // 2. Direct config: {"command": "...", "args": [...], "env": {...}}

          let actualConfig = stdioConfig;

          // If it's the full mcpServers structure, extract the first server config
          if (stdioConfig.mcpServers && typeof stdioConfig.mcpServers === "object") {
            const serverNames = Object.keys(stdioConfig.mcpServers);
            if (serverNames.length > 0) {
              const firstServerName = serverNames[0];
              actualConfig = stdioConfig.mcpServers[firstServerName];

              // If no alias is provided, use the server name from the JSON
              if (!restValues.server_name) {
                restValues.server_name = firstServerName.replace(/-/g, "_"); // Replace hyphens with underscores
              }
            }
          }

          stdioFields = {
            command: actualConfig.command,
            args: actualConfig.args,
            env: actualConfig.env,
          };
        } catch (error) {
          NotificationsManager.fromBackend("Invalid JSON in stdio configuration");
          return;
        }
      }

      // Map "openapi" transport to "http" for the backend
      if (restValues.transport === TRANSPORT.OPENAPI) {
        restValues.transport = "http";
      }

      // Parse token_validation JSON if provided
      let tokenValidation: Record<string, any> | null = null;
      if (rawTokenValidationJson && rawTokenValidationJson.trim() !== "") {
        try {
          tokenValidation = JSON.parse(rawTokenValidationJson);
        } catch {
          NotificationsManager.fromBackend("Invalid JSON in Token Validation Rules");
          setIsLoading(false);
          return;
        }
      }

      // Prepare the payload with cost configuration and allowed tools
      const payload: Record<string, any> = {
        ...restValues,
        ...stdioFields,
        // Remove the raw stdio_config field as we've extracted its components
        stdio_config: undefined,
        mcp_info: {
          server_name: restValues.server_name || restValues.url,
          description: restValues.description,
          logo_url: logoUrl || undefined,
          mcp_server_cost_info: Object.keys(costConfig).length > 0 ? costConfig : null,
          tool_allowlist_enforced: hasToolAllowlistInteraction || allowedTools.length > 0,
        },
        mcp_access_groups: accessGroups,
        alias: restValues.alias,
        allowed_tools: allowedTools,
        tool_name_to_display_name: toolNameToDisplayName,
        tool_name_to_description: toolNameToDescription,
        allow_all_keys: Boolean(allowAllKeysRaw),
        available_on_public_internet: Boolean(availableOnPublicInternetRaw),
        delegate_auth_to_upstream: Boolean(delegateAuthToUpstreamRaw),
        oauth_passthrough: Boolean(oauthPassthroughRaw),
        // ``dcr_bridge`` is only meaningful for the client-forwarded token
        // modes (true_passthrough / oauth_delegate) and defaults on when the
        // toggle is shown; force false for any other auth type so a stale
        // ``true`` is never persisted. Mirrors the sibling flags above.
        dcr_bridge: isClientForwardedTokenMode(restValues.auth_type) ? Boolean(dcrBridgeRaw ?? true) : false,
        ...(restValues.auth_type === AUTH_TYPE.OAUTH2
          ? {
              oauth2_flow:
                values.oauth_flow_type === OAUTH_FLOW.M2M ? MCP_OAUTH2_FLOW_M2M : MCP_OAUTH2_FLOW_INTERACTIVE,
            }
          : {}),
        static_headers: staticHeaders,
        env_vars: envVars,
        ...(tokenValidation !== null && { token_validation: tokenValidation }),
      };

      const includeCredentials =
        restValues.auth_type && AUTH_TYPES_REQUIRING_CREDENTIALS.includes(restValues.auth_type);

      // Client-forwarded rows persist ONLY the declared app; strip any token material that lingered in
      // the form (e.g. from a prior oauth2 authorize on the same session) so it can never reach the row.
      const submitCredentials = isClientForwardedTokenMode(restValues.auth_type)
        ? preservedDeclaredAppCredentials(credentialsPayload)
        : credentialsPayload;

      if (includeCredentials && submitCredentials && Object.keys(submitCredentials).length > 0) {
        payload.credentials = submitCredentials;
      }

      // The selected ID-JAG client-auth method is authoritative: null the other method's
      // fields so the backend's credentials merge cannot keep a stale method alive.
      if (restValues.auth_type === AUTH_TYPE.OAUTH2_ID_JAG) {
        const idJagMethodNulls =
          (idJagClientAuthMethodRaw ?? "client_secret") === "private_key_jwt"
            ? { client_secret: null }
            : { client_private_key: null, client_private_key_id: null, client_assertion_signing_alg: null };
        payload.credentials = { ...(payload.credentials ?? {}), ...idJagMethodNulls };
      }

      // An interactive (oauth2) create persists its DCR-minted client from the ref (kept out of the
      // form store); reuse a re-authorize's registered client instead of re-registering.
      if (restValues.auth_type === AUTH_TYPE.OAUTH2 && dcrClientRef.current) {
        payload.credentials = { ...(payload.credentials ?? {}), ...dcrClientRef.current };
      }

      if (accessToken != null) {
        const response = isAdmin
          ? await createMCPServer(accessToken, payload)
          : await registerMCPServer(accessToken, payload);

        // Persist the token obtained via "Authorize & Fetch" once the server
        // exists (so we have its server_id). OBO holds the per-user token in the
        // backend, so write it to the DB (has_credentials=True). Passthrough
        // forwards a browser-held token, so it stays in sessionStorage only.
        if (oauthTokenResponse?.access_token && response?.server_id) {
          const oauthMode = getMcpOAuthMode({
            auth_type: restValues.auth_type,
            oauth2_flow: values.oauth_flow_type === OAUTH_FLOW.M2M ? MCP_OAUTH2_FLOW_M2M : null,
            delegate_auth_to_upstream: Boolean(delegateAuthToUpstreamRaw),
          });
          if (oauthMode === "authorization_code") {
            const scope = oauthTokenResponse.scope;
            const oauthCredentialPayload = {
              access_token: oauthTokenResponse.access_token,
              refresh_token: oauthTokenResponse.refresh_token,
              expires_in: oauthTokenResponse.expires_in,
              scopes: typeof scope === "string" && scope ? scope.split(" ") : undefined,
            };
            await storeMCPOAuthUserCredential(accessToken, response.server_id, oauthCredentialPayload);
          } else {
            const browserHeldToken = {
              access_token: oauthTokenResponse.access_token,
              expires_in: oauthTokenResponse.expires_in,
              refresh_token: oauthTokenResponse.refresh_token,
              token_type: oauthTokenResponse.token_type,
            };
            setToken(response.server_id, browserHeldToken, userID);
          }
        }

        NotificationsManager.success(
          isAdmin
            ? "MCP Server created successfully"
            : {
                message: "MCP Server submitted for admin review",
                description: "Once an admin approves it, the server will appear in your MCP Servers list.",
              },
        );
        form.resetFields();
        setCostConfig({});
        clearTools();
        setAllowedTools([]);
        setHasToolAllowlistInteraction(false);
        setAliasManuallyEdited(false);
        setLogoUrl(undefined);
        setModalVisible(false);
        onCreateSuccess(response);
      }
    } catch (error) {
      const reason = error instanceof Error ? error.message : String(error);
      NotificationsManager.fromBackend(
        isAdmin ? `Error creating MCP Server: ${reason}` : `Error submitting MCP Server: ${reason}`,
      );
    } finally {
      setIsLoading(false);
    }
  };

  // state
  const handleCancel = () => {
    form.resetFields();
    setCostConfig({});
    clearTools();
    setAllowedTools([]);
    setHasToolAllowlistInteraction(false);
    setAliasManuallyEdited(false);
    setLogoUrl(undefined);
    setAuthorizedIdentity(undefined);
    dcrClientRef.current = null;
    setAppMayNotMatchUpstream(false);
    setModalVisible(false);
  };

  const handleTransportChange = (value: string) => {
    setTransportType(value);
    // Clear fields that are not relevant for the selected transport
    const transportValues =
      value === "stdio"
        ? { url: undefined, spec_path: undefined, auth_type: undefined, credentials: undefined }
        : value === TRANSPORT.OPENAPI
          ? { url: undefined, command: undefined, args: undefined, env: undefined }
          : { spec_path: undefined, command: undefined, args: undefined, env: undefined };

    form.setFieldsValue(transportValues);
    if (isHeldOAuthTokenStale(form.getFieldsValue(true), authorizedIdentity)) {
      clearHeldOAuthToken();
    }
    setFormValues(form.getFieldsValue(true));
  };

  // Generate options with existing groups and potential new group
  const getAccessGroupOptions = () => {
    const existingOptions = availableAccessGroups.map((group: string) => ({
      value: group,
      label: (
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
          <span className="font-medium">{group}</span>
        </div>
      ),
    }));

    // If search value doesn't match any existing group and is not empty, add "create new group" option
    if (
      searchValue &&
      !availableAccessGroups.some((group) => group.toLowerCase().includes(searchValue.toLowerCase()))
    ) {
      existingOptions.push({
        value: searchValue,
        label: (
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
            <span className="font-medium">{searchValue}</span>
            <span className="text-gray-400 text-xs ml-1">create new group</span>
          </div>
        ),
      });
    }

    return existingOptions;
  };

  // Auto-populate alias from server_name unless manually edited
  React.useEffect(() => {
    if (!aliasManuallyEdited && formValues.server_name) {
      const normalized = formValues.server_name.replace(/\s+/g, "_");
      form.setFieldsValue({ alias: normalized });
      setFormValues((prev) => ({ ...prev, alias: normalized }));
    }
  }, [formValues.server_name]);

  // Clear form, tools, and OAuth state when the modal closes so a previous server's
  // authorization, credentials, or tool list never bleed into the next "Add New MCP
  // Server" session, including when a parent dismisses the modal without routing
  // through handleCancel or handleCreate. Only a real open -> closed transition may
  // trigger this: on the post-OAuth-redirect remount the modal starts closed while
  // resumeOAuthFlow's token exchange is in flight, and resetting then discards the
  // fetched token.
  const wasModalVisibleRef = React.useRef(isModalVisible);
  React.useEffect(() => {
    const wasVisible = wasModalVisibleRef.current;
    wasModalVisibleRef.current = isModalVisible;
    if (!isModalVisible && wasVisible) {
      form.resetFields();
      setFormValues({});
      setOauthAccessToken(null);
      clearTools();
      resetOAuthFlow();
      setAuthorizedIdentity(undefined);
      dcrClientRef.current = null;
      setAppMayNotMatchUpstream(false);
    }
  }, [isModalVisible, form, clearTools, resetOAuthFlow]);

  const isAdmin = isAdminRole(userRole);

  const handleFormValuesChange = (changedValues: Record<string, unknown>, allValues: Record<string, unknown>) => {
    // Any change to a mint-relevant field (url, auth_type, oauth_flow_type, client creds/scopes, or the
    // authorization/token/registration endpoints — see getOAuthAuthorizationIdentity) makes a held token
    // stale, so discard it and force a fresh authorize. The stale check reads getFieldsValue(true): the
    // onValuesChange allValues argument holds only MOUNTED paths, so an unmounted identity field (e.g.
    // an oauth_flow_type initialValue while in a client-forwarded mode) would compare as changed on
    // every keystroke and churn the held token. When a clear happens, formValues is rebuilt from the
    // form's post-reset state (not the pre-reset snapshot, which still holds the discarded token).
    // Editing the client fields is the admin managing/acknowledging the app, so it always dismisses
    // the "may not match upstream" warning regardless of the stale-token branch below.
    // Editing the client fields is the admin managing/acknowledging the app, so it dismisses the "may
    // not match upstream" warning. Otherwise a url/endpoint change while a declared app is present keeps
    // the app but flags that it may not match the new upstream (the "keep + warn" behavior). This is
    // independent of the held-token stale check below so it fires even without an authorize this session.
    if ("credentials" in changedValues) {
      setAppMayNotMatchUpstream(false);
    } else {
      const upstreamChanged = ["url", "spec_path", "issuer", "authorization_url", "token_url", "registration_url"].some(
        (key) => key in changedValues,
      );
      const hasDeclaredApp = preservedDeclaredAppCredentials(form.getFieldValue("credentials")) !== undefined;
      if (upstreamChanged && hasDeclaredApp) {
        setAppMayNotMatchUpstream(true);
      }
    }
    if (isHeldOAuthTokenStale(form.getFieldsValue(true), authorizedIdentity)) {
      clearHeldOAuthToken(changedValues);
      setFormValues(form.getFieldsValue(true));
      return;
    }
    setFormValues(allValues);
  };

  // rendering
  return (
    <Modal
      title={
        <div className="flex items-center pb-4 border-b border-gray-100" style={{ gap: 12 }}>
          {onBackToDiscovery && (
            <button
              onClick={onBackToDiscovery}
              className="text-sm text-blue-600 hover:text-blue-800 cursor-pointer bg-transparent border-none"
              style={{ flexShrink: 0 }}
            >
              &#8592;
            </button>
          )}
          <img
            src={resolveLogoSrc(mcpLogoImg)}
            alt="MCP Logo"
            className="w-8 h-8 object-contain"
            style={{
              height: "20px",
              width: "20px",
              objectFit: "contain",
            }}
          />
          <h2 className="text-xl font-semibold text-gray-900">
            {isAdmin ? "Add New MCP Server" : "Submit MCP Server for Review"}
          </h2>
        </div>
      }
      open={isModalVisible}
      width={1000}
      onCancel={handleCancel}
      footer={null}
      forceRender
      className="top-8"
      styles={{
        body: { padding: "24px" },
        header: { padding: "24px 24px 0 24px", border: "none" },
      }}
    >
      <div className="mt-6">
        <Form
          form={form}
          onFinish={handleCreate}
          onValuesChange={handleFormValuesChange}
          layout="vertical"
          className="space-y-6"
        >
          {!isAdmin && (
            <div className="rounded-md bg-blue-50 border border-blue-200 px-4 py-3 text-sm text-blue-800">
              Your submission will be sent for admin review. Once approved, the server will appear in your MCP Servers
              list. The request must be made with a team-scoped API key.
            </div>
          )}
          <div className="grid grid-cols-1 gap-6">
            <Form.Item
              label={
                <span className="text-sm font-medium text-gray-700 flex items-center">
                  MCP Server Name
                  <Tooltip title="Best practice: Use a descriptive name that indicates the server's purpose (e.g., 'GitHub_MCP', 'Email_Service'). Cannot contain spaces or hyphens; use underscores instead. Names must comply with SEP-986 and will be rejected if invalid (https://modelcontextprotocol.io/specification/2025-11-25/server/tools#tool-names).">
                    <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                  </Tooltip>
                </span>
              }
              name="server_name"
              rules={[
                { required: false, message: "Please enter a server name" },
                { validator: (_, value) => validateMCPServerName(value) },
              ]}
            >
              <TextInput
                placeholder="e.g., GitHub_MCP, Zapier_MCP, etc."
                className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
              />
            </Form.Item>

            <Form.Item
              label={
                <span className="text-sm font-medium text-gray-700 flex items-center">
                  Alias
                  <Tooltip title="A short, unique identifier for this server. Defaults to the server name if not provided. Cannot contain spaces or hyphens; use underscores instead.">
                    <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                  </Tooltip>
                </span>
              }
              name="alias"
              rules={[{ required: false }, { validator: (_, value) => validateMCPServerName(value) }]}
            >
              <TextInput
                placeholder="e.g., GitHub_MCP, Zapier_MCP, etc."
                className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                onChange={() => setAliasManuallyEdited(true)}
              />
            </Form.Item>

            <Form.Item
              label={<span className="text-sm font-medium text-gray-700">Description</span>}
              name="description"
              rules={[
                {
                  required: false,
                  message: "Please enter a server description",
                },
              ]}
            >
              <TextInput
                placeholder="Brief description of what this server does"
                className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
              />
            </Form.Item>

            <MCPLogoSelector value={logoUrl} onChange={setLogoUrl} />

            <Form.Item
              label={<span className="text-sm font-medium text-gray-700">GitHub / Source URL</span>}
              name="source_url"
            >
              <TextInput
                placeholder="https://github.com/org/mcp-server"
                className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
              />
            </Form.Item>

            <Form.Item
              label={<span className="text-sm font-medium text-gray-700">Transport Type</span>}
              name="transport"
              rules={[{ required: true, message: "Please select a transport type" }]}
            >
              <Select
                placeholder="Select transport"
                className="rounded-lg"
                size="large"
                onChange={handleTransportChange}
                value={transportType}
              >
                <Select.Option value="http">Streamable HTTP (Recommended)</Select.Option>
                <Select.Option value="sse">Server-Sent Events (SSE)</Select.Option>
                <Select.Option value="stdio">Standard Input/Output (stdio)</Select.Option>
                <Select.Option value={TRANSPORT.OPENAPI}>OpenAPI Spec</Select.Option>
              </Select>
            </Form.Item>

            {/* URL field - only show for HTTP and SSE */}
            {(transportType === "http" || transportType === "sse") && (
              <Form.Item
                label={<span className="text-sm font-medium text-gray-700">MCP Server URL</span>}
                name="url"
                rules={[
                  { required: true, message: "Please enter a server URL" },
                  { validator: (_, value) => validateMCPServerUrl(value) },
                ]}
              >
                <Input
                  placeholder="https://your-mcp-server.com"
                  className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                />
              </Form.Item>
            )}

            {/* OpenAPI: logo picker + spec URL input */}
            {transportType === TRANSPORT.OPENAPI && (
              <OpenAPIFormSection
                form={form}
                accessToken={isModalVisible ? accessToken : null}
                onValuesChange={(updates) =>
                  handleFormValuesChange(updates, { ...form.getFieldsValue(true), ...updates })
                }
                onKeyToolsChange={setKeyTools}
                onLogoUrlChange={setLogoUrl}
                onOAuthDocsUrlChange={setOauthDocsUrl}
              />
            )}

            {/* BYOK toggle - only for OpenAPI */}
            {transportType === TRANSPORT.OPENAPI && (
              <>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center gap-2">
                      BYOK (Bring Your Own Key)
                      <Tooltip title="When enabled, each user provides their own API key for this service. Keys are stored per-user and never shared.">
                        <InfoCircleOutlined className="text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name="is_byok"
                  valuePropName="checked"
                >
                  <Switch />
                </Form.Item>

                <Form.Item
                  noStyle
                  shouldUpdate={(prev, cur) => prev.is_byok !== cur.is_byok || prev.auth_type !== cur.auth_type}
                >
                  {({ getFieldValue }) =>
                    getFieldValue("is_byok") ? (
                      <>
                        {/* Auth format hint */}
                        {getFieldValue("auth_type") && getFieldValue("auth_type") !== "none" && (
                          <div className="mb-4 p-3 bg-blue-50 rounded-lg text-sm text-blue-700 flex items-start gap-2">
                            <InfoCircleOutlined className="mt-0.5 shrink-0" />
                            <span>
                              User keys will be sent as:{" "}
                              <code className="font-mono bg-blue-100 px-1 rounded-sm">
                                {getFieldValue("auth_type") === "bearer_token" && "Authorization: Bearer {key}"}
                                {getFieldValue("auth_type") === "token" && "Authorization: token {key}"}
                                {getFieldValue("auth_type") === "api_key" && "x-api-key: {key}"}
                                {getFieldValue("auth_type") === "basic" && "Authorization: Basic {key}"}
                                {getFieldValue("auth_type") === "authorization" && "Authorization: {key}"}
                              </code>
                              {!getFieldValue("auth_type") && "Set Authentication Type below to specify the format."}
                            </span>
                          </div>
                        )}
                        {!getFieldValue("auth_type") && (
                          <div className="mb-4 p-3 bg-yellow-50 rounded-lg text-sm text-yellow-700 flex items-start gap-2">
                            <InfoCircleOutlined className="mt-0.5 shrink-0" />
                            <span>
                              Set the <strong>Authentication Type</strong> below to specify how user keys are sent
                              (e.g., Bearer Token, API Key header).
                            </span>
                          </div>
                        )}
                        <Form.Item
                          label={
                            <span className="text-sm font-medium text-gray-700">
                              Access Description
                              <Tooltip title="List of permissions shown to users in the connection modal (e.g. 'Create and manage Jira issues')">
                                <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                              </Tooltip>
                            </span>
                          }
                          name="byok_description"
                        >
                          <Select
                            mode="tags"
                            placeholder="Add access description items (press Enter after each)"
                            className="w-full"
                            tokenSeparators={[","]}
                          />
                        </Form.Item>

                        <Form.Item
                          label={
                            <span className="text-sm font-medium text-gray-700">
                              API Key Help URL
                              <Tooltip title="Optional link shown to users to help them find their API key">
                                <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                              </Tooltip>
                            </span>
                          }
                          name="byok_api_key_help_url"
                        >
                          <Input placeholder="https://docs.example.com/api-keys" />
                        </Form.Item>
                      </>
                    ) : null
                  }
                </Form.Item>
              </>
            )}

            <Form.Item
              label={
                <span className="text-sm font-medium text-gray-700 flex items-center">
                  Max Concurrent Requests (optional)
                  <Tooltip title="Maximum number of tool calls LiteLLM will run against this server at the same time. Additional calls wait for a free slot. Leave blank for no limit.">
                    <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                  </Tooltip>
                </span>
              }
              name="max_concurrent_requests"
            >
              <InputNumber
                min={1}
                precision={0}
                placeholder="e.g. 10"
                style={{ width: "100%" }}
                className="rounded-lg"
              />
            </Form.Item>

            {/* Authentication - show for HTTP, SSE, and OpenAPI */}
            {transportType !== "stdio" && transportType !== "" && (
              <Collapse
                defaultActiveKey={["auth"]}
                className="mb-4"
                items={[
                  {
                    key: "auth",
                    label: <span className="text-sm font-semibold text-gray-700">Authentication</span>,
                    children: (
                      <>
                        <Form.Item name="auth_type" rules={[{ required: true, message: "Please select an auth type" }]}>
                          <Select placeholder="Select auth type" className="rounded-lg" size="large" virtual={false}>
                            <Select.Option value="none">None</Select.Option>
                            <Select.Option value="api_key">API Key</Select.Option>
                            <Select.Option value="bearer_token">Bearer Token</Select.Option>
                            <Select.Option value="token">Token</Select.Option>
                            <Select.Option value="basic">Basic Auth</Select.Option>
                            <Select.Option value="oauth2">OAuth</Select.Option>
                            <Select.Option value="oauth2_token_exchange">OAuth Token Exchange (OBO)</Select.Option>
                            <Select.Option value="oauth2_id_jag">
                              Enterprise-Managed Authorization (ID-JAG)
                            </Select.Option>
                            <Select.Option value="aws_sigv4">AWS SigV4 (Bedrock AgentCore MCPs)</Select.Option>
                            <Select.Option value="true_passthrough">True Passthrough (no LiteLLM auth)</Select.Option>
                            <Select.Option value="oauth_delegate">
                              OAuth Delegate (client-supplied upstream token)
                            </Select.Option>
                          </Select>
                        </Form.Item>

                        <TruePassthroughWarning authType={authType} />

                        <PassthroughAuthorizeSection
                          authType={authType}
                          dcrBridgeInitialChecked
                          oauthFlow={{
                            startOAuthFlow,
                            status: oauthStatus,
                            error: oauthError,
                            tokenResponse: oauthTokenResponse,
                          }}
                          appMayNotMatchUpstream={appMayNotMatchUpstream}
                        />

                        {shouldShowAuthValueField && (
                          <Form.Item
                            label={
                              <span className="text-sm font-medium text-gray-700 flex items-center">
                                Authentication Value
                                <Tooltip title="Token, password, or header value to send with each request for the selected auth type.">
                                  <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                                </Tooltip>
                              </span>
                            }
                            name={["credentials", "auth_value"]}
                            rules={[
                              {
                                validator: (_, value) =>
                                  value && typeof value === "string" && value.trim() === ""
                                    ? Promise.reject(new Error("Authentication value cannot be empty whitespace"))
                                    : Promise.resolve(),
                              },
                            ]}
                          >
                            <TextInput
                              type="password"
                              placeholder="Enter token or secret"
                              className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                            />
                          </Form.Item>
                        )}

                        {isOAuthAuthType && (
                          <OAuthFormFields
                            isM2M={isM2MFlow}
                            initialFlowType={OAUTH_FLOW.INTERACTIVE}
                            docsUrl={oauthDocsUrl}
                            oauthFlow={{
                              startOAuthFlow,
                              status: oauthStatus,
                              error: oauthError,
                              tokenResponse: oauthTokenResponse,
                            }}
                          />
                        )}

                        {isTokenExchangeAuthType && <TokenExchangeFormFields />}

                        {isIdJagAuthType && <IdJagFormFields />}
                      </>
                    ),
                  },
                ]}
              />
            )}

            {transportType !== "stdio" && transportType !== "" && isAwsSigV4AuthType && (
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
                  <Input
                    placeholder="us-east-1"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
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
                          return Promise.reject(
                            new Error("Access Key ID is required when Secret Access Key is provided"),
                          );
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
                          return Promise.reject(
                            new Error("Secret Access Key is required when Access Key ID is provided"),
                          );
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
            )}

            {/* Stdio Configuration - only show for stdio transport */}
            <StdioConfiguration isVisible={transportType === "stdio"} />
          </div>

          {/* Environment Variables Section */}
          <div className="mt-8">
            <EnvVarsSection />
          </div>

          {/* Permission Management / Access Control Section */}
          <div className="mt-8">
            <MCPPermissionManagement
              availableAccessGroups={availableAccessGroups}
              mcpServer={null}
              searchValue={searchValue}
              setSearchValue={setSearchValue}
              getAccessGroupOptions={getAccessGroupOptions}
            />
          </div>

          {/* Connection Status Section */}
          <div className="mt-8 pt-6 border-t border-gray-200">
            <MCPConnectionStatus
              formValues={formValues}
              tools={tools}
              isLoadingTools={isLoadingTools}
              toolsError={toolsError}
              toolsErrorStatus={toolsErrorStatus}
              toolsErrorStackTrace={toolsErrorStackTrace}
              canFetchTools={canFetchTools}
              fetchTools={fetchTools}
            />
          </div>

          {/* Tool Configuration Section */}
          <div className="mt-6">
            <MCPToolConfiguration
              accessToken={accessToken}
              formValues={formValues}
              allowedTools={allowedTools}
              existingAllowedTools={null}
              onAllowedToolsChange={setAllowedTools}
              hasToolAllowlistInteraction={hasToolAllowlistInteraction}
              onToolAllowlistInteraction={() => setHasToolAllowlistInteraction(true)}
              toolNameToDisplayName={toolNameToDisplayName}
              toolNameToDescription={toolNameToDescription}
              onToolNameToDisplayNameChange={setToolNameToDisplayName}
              onToolNameToDescriptionChange={setToolNameToDescription}
              keyTools={keyTools}
              externalTools={tools}
              externalIsLoading={isLoadingTools}
              externalError={toolsError}
              externalErrorStatus={toolsErrorStatus}
              externalCanFetch={canFetchTools}
            />
          </div>

          {/* Cost Configuration Section */}
          <div className="mt-6">
            <MCPServerCostConfig
              value={costConfig}
              onChange={setCostConfig}
              tools={tools.filter((tool) => allowedTools.includes(tool.name))}
              disabled={false}
            />
          </div>

          <div className="flex items-center justify-end space-x-3 pt-6 border-t border-gray-100">
            <Button variant="secondary" onClick={handleCancel}>
              Cancel
            </Button>
            <Button variant="primary" loading={isLoading}>
              {isLoading ? "Creating..." : "Add MCP Server"}
            </Button>
          </div>
        </Form>
      </div>
    </Modal>
  );
};

export default CreateMCPServer;
