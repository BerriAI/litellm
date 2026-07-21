import React, { useState, useEffect } from "react";
import { Form, Select, Button as AntdButton, Tooltip, Input, InputNumber, Alert } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Button, TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import {
  AUTH_TYPE,
  isClientForwardedTokenMode,
  getOAuthAuthorizationIdentity,
  CLEARED_ON_INVALIDATION,
  isHeldOAuthTokenStale,
  preservedDeclaredAppCredentials,
  withoutMintedTokenCredentials,
  OAUTH_FLOW,
  MCP_OAUTH2_FLOW_M2M,
  MCP_OAUTH2_FLOW_INTERACTIVE,
  MCPServer,
  MCPServerCostInfo,
  TRANSPORT,
  getMcpOAuthMode,
  oauth2FlowToFormValue,
} from "@/components/mcp_tools/types";
import {
  updateMCPServer,
  listMCPTools,
  storeMCPOAuthUserCredential,
  testMCPToolsListRequest,
} from "@/components/networking";
import { getToken, isTokenValid, removeToken, setToken } from "@/utils/mcpTokenStore";
import { buildMcpPassthroughAuthHeader } from "@/utils/mcpHeaderUtils";
import MCPServerCostConfig from "./mcp_server_cost_config";
import MCPPermissionManagement from "./MCPPermissionManagement";
import TruePassthroughWarning from "./TruePassthroughWarning";
import PassthroughAuthorizeSection from "./PassthroughAuthorizeSection";
import MCPToolConfiguration from "./mcp_tool_configuration";
import StdioConfiguration from "./StdioConfiguration";
import TokenExchangeFormFields from "./TokenExchangeFormFields";
import MCPLogoSelector from "./MCPLogoSelector";
import EnvVarsSection from "./EnvVarsSection";
import TokenEndpointAuthMethodField from "./TokenEndpointAuthMethodField";
import {
  validateMCPServerUrl,
  validateMCPServerName,
  normalizeEnvVars,
  normalizeToolOverrideMap,
  TOOL_DISPLAY_NAME_PATTERN,
} from "./utils";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { useMcpOAuthFlow } from "@/hooks/useMcpOAuthFlow";
import { getSecureItem, setSecureItem } from "@/utils/secureStorage";

interface MCPServerEditProps {
  mcpServer: MCPServer;
  accessToken: string | null;
  userID?: string | null;
  onCancel: () => void;
  onSuccess: (server: MCPServer) => void;
  availableAccessGroups: string[];
}

const AUTH_TYPES_REQUIRING_AUTH_VALUE = [AUTH_TYPE.API_KEY, AUTH_TYPE.BEARER_TOKEN, AUTH_TYPE.TOKEN, AUTH_TYPE.BASIC];
const AUTH_TYPES_REQUIRING_CREDENTIALS = [
  ...AUTH_TYPES_REQUIRING_AUTH_VALUE,
  AUTH_TYPE.OAUTH2,
  AUTH_TYPE.OAUTH2_TOKEN_EXCHANGE,
  AUTH_TYPE.AWS_SIGV4,
  AUTH_TYPE.TRUE_PASSTHROUGH,
  AUTH_TYPE.OAUTH_DELEGATE,
];
export const EDIT_OAUTH_UI_STATE_KEY = "litellm-mcp-oauth-edit-state";

const MCPServerEdit: React.FC<MCPServerEditProps> = ({
  mcpServer,
  accessToken,
  userID,
  onCancel,
  onSuccess,
  availableAccessGroups,
}) => {
  const [form] = Form.useForm();
  const [costConfig, setCostConfig] = useState<MCPServerCostInfo>({});
  const [tools, setTools] = useState<any[]>([]);
  const [isLoadingTools, setIsLoadingTools] = useState(false);
  const [toolsError, setToolsError] = useState<string | null>(null);
  const [searchValue, setSearchValue] = useState<string>("");
  const [aliasManuallyEdited, setAliasManuallyEdited] = useState(false);
  const [removeStoredApp, setRemoveStoredApp] = useState(false);
  // Set when the upstream identity (url/endpoints) changed while a declared app is present, so the
  // section warns that the saved app may not match the new upstream (the app is kept, not wiped).
  const [appMayNotMatchUpstream, setAppMayNotMatchUpstream] = useState(false);
  const [allowedTools, setAllowedTools] = useState<string[]>([]);
  const [hasToolAllowlistInteraction, setHasToolAllowlistInteraction] = useState(false);
  const [toolNameToDisplayName, setToolNameToDisplayName] = useState<Record<string, string>>({});
  const [toolNameToDescription, setToolNameToDescription] = useState<Record<string, string>>({});
  const [pendingRestoredValues, setPendingRestoredValues] = useState<Record<string, any> | null>(null);
  const [logoUrl, setLogoUrl] = useState<string | undefined>(mcpServer.mcp_info?.logo_url || undefined);
  const authType = Form.useWatch("auth_type", form) as string | undefined;
  const transportType = Form.useWatch("transport", form) as string | undefined;
  const isStdioTransport = transportType === "stdio";
  const isOpenAPITransport = transportType === TRANSPORT.OPENAPI;
  const isMCPTransport = !isStdioTransport && !isOpenAPITransport;
  const shouldShowAuthValueField = authType ? AUTH_TYPES_REQUIRING_AUTH_VALUE.includes(authType) : false;
  const isOAuthAuthType = authType === AUTH_TYPE.OAUTH2;
  const isTokenExchangeAuthType = authType === AUTH_TYPE.OAUTH2_TOKEN_EXCHANGE;
  const isAwsSigV4AuthType = authType === AUTH_TYPE.AWS_SIGV4;
  const oauthFlowTypeValue = Form.useWatch("oauth_flow_type", form) as string | undefined;
  const isM2MFlow = isOAuthAuthType && oauthFlowTypeValue === OAUTH_FLOW.M2M;
  // Watch reflects a live toggle when the delegate switch is mounted; fall back to
  // the stored value otherwise (useWatch returns undefined for an unmounted field,
  // the same trap the oauth_flow_type field originally hit).
  const delegateAuthWatched = Form.useWatch("delegate_auth_to_upstream", form) as boolean | undefined;
  const isDelegateAuth = delegateAuthWatched ?? Boolean(mcpServer.delegate_auth_to_upstream);

  // Watch form fields that affect tool fetching
  const currentUrl = Form.useWatch("url", form);
  const currentSpecPath = Form.useWatch("spec_path", form);
  const currentServerName = Form.useWatch("server_name", form);
  const currentAuthType = Form.useWatch("auth_type", form);
  const currentStaticHeaders = Form.useWatch("static_headers", form);
  const currentCredentials = Form.useWatch("credentials", form);
  const currentIssuer = Form.useWatch("issuer", form);
  const currentAuthorizationUrl = Form.useWatch("authorization_url", form);
  const currentTokenUrl = Form.useWatch("token_url", form);
  const currentRegistrationUrl = Form.useWatch("registration_url", form);
  const hasExistingToolAllowlist =
    Boolean(mcpServer.mcp_info?.tool_allowlist_enforced) || (mcpServer.allowed_tools?.length ?? 0) > 0;
  const existingAllowedTools = hasExistingToolAllowlist ? mcpServer.allowed_tools ?? [] : null;

  const persistEditUiState = () => {
    if (typeof window === "undefined") {
      return;
    }
    try {
      const values = form.getFieldsValue(true);
      setSecureItem(
        EDIT_OAUTH_UI_STATE_KEY,
        JSON.stringify({
          serverId: mcpServer.server_id,
          formValues: values,
          costConfig,
          allowedTools,
          hasToolAllowlistInteraction,
          searchValue,
          aliasManuallyEdited,
        }),
      );
    } catch (err) {
      console.warn("Failed to persist MCP edit state", err);
    }
  };

  // The auth mode every decision must key off: the admin's in-flight form selection wins over the
  // saved record, so authorizing, loading tools, and saving all agree with what the form shows. Paths
  // that read only mcpServer.auth_type go stale the moment the admin switches modes in the form.
  const getEffectiveAuthType = () => form.getFieldValue("auth_type") ?? mcpServer.auth_type;

  // The OAuth authorization identity (see getOAuthAuthorizationIdentity) captured when a token is fetched
  // in this edit session; undefined when none is held. If a mint-relevant field later diverges from it,
  // the held token (hook response + sessionStorage) is discarded so the admin must re-authorize.
  const authorizedIdentityRef = React.useRef<string | undefined>(undefined);

  const {
    startOAuthFlow,
    status: oauthStatus,
    error: oauthError,
    tokenResponse: oauthTokenResponse,
    reset: resetOAuthFlow,
  } = useMcpOAuthFlow({
    accessToken,
    getCredentials: () => form.getFieldValue("credentials"),
    getTemporaryPayload: () => {
      const values = form.getFieldsValue(true);
      const url = values.url || mcpServer.url;
      const transport = values.transport || mcpServer.transport;
      if (!url || !transport) {
        return null;
      }
      const staticHeaders = Array.isArray(values.static_headers)
        ? values.static_headers.reduce((acc: Record<string, string>, entry: Record<string, string>) => {
            const header = entry?.header?.trim();
            if (!header) {
              return acc;
            }
            acc[header] = (entry?.value ?? "").trim();
            return acc;
          }, {})
        : ({} as Record<string, string>);

      return {
        server_id: mcpServer.server_id,
        server_name: values.server_name || mcpServer.server_name || mcpServer.alias,
        alias: values.alias || mcpServer.alias,
        description: values.description || mcpServer.description,
        url,
        transport,
        auth_type: isClientForwardedTokenMode(values.auth_type) ? values.auth_type : AUTH_TYPE.OAUTH2,
        credentials: isClientForwardedTokenMode(values.auth_type)
          ? preservedDeclaredAppCredentials(values.credentials)
          : values.credentials,
        mcp_access_groups: values.mcp_access_groups || mcpServer.mcp_access_groups,
        static_headers: staticHeaders,
        command: values.command,
        args: values.args,
        env: values.env,
      };
    },
    onTokenReceived: (token) => {
      if (!token?.access_token) {
        return;
      }

      authorizedIdentityRef.current = getOAuthAuthorizationIdentity(form.getFieldsValue(true));
      if (isClientForwardedTokenMode(getEffectiveAuthType())) {
        const browserHeldToken = {
          access_token: token.access_token,
          expires_in: token.expires_in,
          refresh_token: token.refresh_token,
          token_type: token.token_type,
        };
        setToken(mcpServer.server_id, browserHeldToken, userID);
        NotificationsManager.success(
          "Token held for this browser session. Tools can now be loaded and configured; the token is not saved to LiteLLM.",
        );
        return;
      }

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
      // siblings behind; the admin-typed client keys and scopes are carried explicitly above.
      form.setFieldValue("credentials", nextCredentials);
      // Re-capture after writing credentials so the token is not invalidated by its own credential write.
      authorizedIdentityRef.current = getOAuthAuthorizationIdentity(form.getFieldsValue(true));

      NotificationsManager.success(
        "OAuth authorization successful! Please click 'Update MCP Server' to save the credentials.",
      );
    },
    onBeforeRedirect: persistEditUiState,
    flowSource: "edit",
  });

  const initialStaticHeaders = React.useMemo(() => {
    if (!mcpServer.static_headers) {
      return [];
    }
    return Object.entries(mcpServer.static_headers).map(([header, value]) => ({
      header,
      value: value != null ? String(value) : "",
    }));
  }, [mcpServer.static_headers]);

  const initialEnvVars = React.useMemo(() => {
    if (!Array.isArray(mcpServer.env_vars)) {
      return [];
    }
    return mcpServer.env_vars.map((entry) => ({
      name: entry.name,
      value: entry.value ?? "",
      scope: entry.scope === "user" ? "user" : "global",
      description: entry.description ?? "",
    }));
  }, [mcpServer.env_vars]);

  const initialEnvJson = React.useMemo(() => {
    const env = mcpServer.env ?? undefined;
    if (!env || Object.keys(env).length === 0) {
      return "";
    }
    try {
      return JSON.stringify(env, null, 2);
    } catch {
      return "";
    }
  }, [mcpServer.env]);

  // If server has spec_path, show it as "openapi" transport in the UI
  const effectiveTransport = React.useMemo(() => {
    if (mcpServer.spec_path && mcpServer.transport !== "stdio") {
      return TRANSPORT.OPENAPI;
    }
    return mcpServer.transport;
  }, [mcpServer]);

  const initialValues = React.useMemo(
    () => ({
      ...mcpServer,
      transport: effectiveTransport,
      static_headers: initialStaticHeaders,
      env_vars: initialEnvVars,
      extra_headers: mcpServer.extra_headers || [],
      oauth_flow_type: oauth2FlowToFormValue(mcpServer.oauth2_flow),
      dcr_bridge: Boolean(mcpServer.dcr_bridge),
      token_validation_json: mcpServer.token_validation
        ? JSON.stringify(mcpServer.token_validation, null, 2)
        : undefined,
    }),
    [mcpServer, effectiveTransport, initialStaticHeaders, initialEnvVars, initialEnvJson],
  );

  // antd applies `initialValues` only at first mount. When the server loads after
  // mount (e.g. returning from the OAuth redirect lands on Overview and the form
  // mounts before the server data is ready), the form would stay blank. Re-sync it
  // from the loaded server once per server_id so it always reflects the saved config;
  // the OAuth-restore effect below then overlays any in-progress edits on top.
  const syncedServerIdRef = React.useRef<string | null>(null);
  useEffect(() => {
    if (!mcpServer.server_id || syncedServerIdRef.current === mcpServer.server_id) {
      return;
    }
    syncedServerIdRef.current = mcpServer.server_id;
    form.setFieldsValue(initialValues);
    // Reset per-server OAuth UI state so it never carries across a server switch without an unmount: a
    // stale removeStoredApp would send an explicit-null credential write that deletes the new server's
    // stored app, and a stale warning would show on a server whose upstream did not change.
    setAppMayNotMatchUpstream(false);
    setRemoveStoredApp(false);
  }, [mcpServer.server_id, initialValues, form]);

  // Initialize cost config from existing server data
  useEffect(() => {
    if (mcpServer.mcp_info?.mcp_server_cost_info) {
      setCostConfig(mcpServer.mcp_info.mcp_server_cost_info);
    }
  }, [mcpServer]);

  // Initialize allowed tools and tool overrides from existing server data
  useEffect(() => {
    setHasToolAllowlistInteraction(false);
  }, [mcpServer.server_id]);

  useEffect(() => {
    if (hasExistingToolAllowlist) {
      setAllowedTools(mcpServer.allowed_tools ?? []);
    }
    setToolNameToDisplayName(normalizeToolOverrideMap(mcpServer.tool_name_to_display_name));
    setToolNameToDescription(normalizeToolOverrideMap(mcpServer.tool_name_to_description));
  }, [mcpServer, hasExistingToolAllowlist]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const storedState = getSecureItem(EDIT_OAUTH_UI_STATE_KEY);
    if (!storedState) {
      return;
    }

    try {
      const parsed = JSON.parse(storedState);
      if (!parsed || parsed.serverId !== mcpServer.server_id) {
        return;
      }
      if (parsed.formValues) {
        // Rebuild credentials from the declared app in EITHER the loaded server or the saved snapshot,
        // then strip minted token material. Merging the two (server under snapshot) before stripping is
        // what guarantees a token-only snapshot never clears a stored client_id/client_secret: the
        // server's declared app survives and only the token keys drop. Assigning the cleaned result (not
        // spreading the raw snapshot) also ensures a stale token can never rehydrate into the form.
        const restoredCredentials = withoutMintedTokenCredentials({
          ...(mcpServer.credentials ?? {}),
          ...((parsed.formValues.credentials as Record<string, unknown> | undefined) ?? {}),
        });
        const restoredValues = {
          ...mcpServer,
          ...parsed.formValues,
          credentials: restoredCredentials,
        };
        setPendingRestoredValues(restoredValues);
      }
      // The ref is re-armed by onTokenReceived when the redirect completes the code exchange, so there
      // is no separate restore-side re-arm here (writing a ref inside an effect is disallowed).
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
    } catch (err) {
      console.error("Failed to restore MCP edit state", err);
    } finally {
      window.sessionStorage.removeItem(EDIT_OAUTH_UI_STATE_KEY);
    }
  }, [form, mcpServer]);

  useEffect(() => {
    if (!pendingRestoredValues) {
      return;
    }
    // Set transport first so transport-dependent fields render, then apply the rest
    // on the re-run triggered by the transportType watch (without it the effect's
    // deps never change and the second pass never runs, leaving fields blank).
    const transport = pendingRestoredValues.transport || mcpServer.transport;
    if (transport && transport !== form.getFieldValue("transport")) {
      form.setFieldsValue({ transport });
      return;
    }
    form.setFieldsValue(pendingRestoredValues);
    setPendingRestoredValues(null);
  }, [pendingRestoredValues, form, mcpServer.transport, transportType]);

  // Transform string array to object array for initial form values
  useEffect(() => {
    if (mcpServer.mcp_access_groups) {
      // If access groups are objects, extract the name property; if strings, use as is
      const groupNames = mcpServer.mcp_access_groups.map((g: any) => (typeof g === "string" ? g : g.name || String(g)));
      form.setFieldValue("mcp_access_groups", groupNames);
    }
  }, [mcpServer]);

  // Fetch tools when component mounts for a saved server
  useEffect(() => {
    if (!mcpServer.server_id || mcpServer.server_id.trim() === "") {
      return;
    }
    fetchTools();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mcpServer, accessToken, userID, oauthTokenResponse?.access_token]);

  // Invalidate a token authorized in this edit session once any mint-relevant field diverges from the
  // identity it was minted against (url, auth_type, oauth_flow_type, client creds/scopes, or the
  // authorization/token/registration endpoints — see getOAuthAuthorizationIdentity). Discards the hook
  // token (resetOAuthFlow, which re-runs fetchTools to prompt a fresh authorize), the sessionStorage
  // token (removeToken, browser-held modes), and the fetched token/DCR client in the shared
  // CLEARED_ON_INVALIDATION form fields; the admin's in-flight edit is re-applied so it is never wiped.
  // Only fires when a token was actually authorized here (ref set), so a token already valid for the
  // saved server on mount is left untouched. Driven from onValuesChange for user input, plus an explicit
  // recheck after programmatic setFieldsValue paths (handleTransportChange), which antd does not report
  // through onValuesChange.
  const clearHeldOAuthToken = (changedValues: Record<string, unknown> = {}) => {
    authorizedIdentityRef.current = undefined;
    if (mcpServer.server_id) {
      removeToken(mcpServer.server_id, userID);
    }
    setTools([]);
    resetOAuthFlow();
    // The admin-typed app is upstream-scoped config, not minted material, so it survives every
    // invalidation; only the held token is discarded. Token-shaped keys are excluded by the filter.
    const keptAppCredentials = preservedDeclaredAppCredentials(form.getFieldValue("credentials"));
    form.resetFields([...CLEARED_ON_INVALIDATION]);
    if (keptAppCredentials) {
      form.setFieldsValue({ credentials: keptAppCredentials });
    }
    const preserved = Object.fromEntries(
      CLEARED_ON_INVALIDATION.filter((key) => key in changedValues).map((key) => [key, changedValues[key]]),
    );
    if (Object.keys(preserved).length > 0) {
      form.setFieldsValue(preserved);
    }
  };

  const handleFormValuesChange = (changedValues: Record<string, unknown>) => {
    // Editing the client fields dismisses the "may not match upstream" warning; otherwise a url/endpoint
    // change while a declared app is present keeps the app but flags that it may not match the new
    // upstream (the "keep + warn" behavior). Mirrors the create form; independent of the held-token
    // stale check so it fires even without an authorize this session (the stored app is for the old url).
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
    if (isHeldOAuthTokenStale(form.getFieldsValue(true), authorizedIdentityRef.current)) {
      clearHeldOAuthToken(changedValues);
    }
  };

  // A token authorized in this edit session for interactive OAuth (authorization_code) is only
  // committed to the DB on save, so a plain by-server_id listing cannot use it and the preview would
  // stay empty until the admin saves; the create form previews the identical state through the
  // config-based preview endpoint, which takes the staged token explicitly. Returns false when there
  // is no staged interactive token so fetchTools falls through to the by-server_id listing.
  const previewWithStagedInteractiveToken = async (
    isPassthrough: boolean,
    isBrowserHeldTokenMode: boolean,
  ): Promise<boolean> => {
    const stagedToken =
      !isPassthrough && !isBrowserHeldTokenMode && getEffectiveAuthType() === AUTH_TYPE.OAUTH2
        ? oauthTokenResponse?.access_token
        : undefined;
    if (!stagedToken) {
      return false;
    }
    setIsLoadingTools(true);
    setToolsError(null);
    try {
      const values = form.getFieldsValue(true);
      const rawTransport = values.transport || mcpServer.transport;
      // oauth2_flow must be explicit: the preview endpoint infers client_credentials from the
      // inherited client_id/client_secret/token_url (common once DCR or discovery filled them) and
      // would strip the staged bearer to preview as M2M. spec_path keeps OpenAPI servers on the
      // spec-based preview path, mirroring the create form's config.
      const previewConfig = {
        server_id: mcpServer.server_id,
        server_name: values.server_name || mcpServer.server_name || mcpServer.alias,
        url: values.url || mcpServer.url,
        spec_path: values.spec_path || mcpServer.spec_path,
        transport: rawTransport === TRANSPORT.OPENAPI ? TRANSPORT.HTTP : rawTransport,
        auth_type: AUTH_TYPE.OAUTH2,
        oauth2_flow: MCP_OAUTH2_FLOW_INTERACTIVE,
        issuer: values.issuer,
        authorization_url: values.authorization_url,
        token_url: values.token_url,
        registration_url: values.registration_url,
      };
      const toolsResponse = await testMCPToolsListRequest(accessToken, previewConfig, stagedToken);
      if (toolsResponse.tools && !toolsResponse.error) {
        setTools(toolsResponse.tools);
      } else {
        setTools([]);
        setToolsError(toolsResponse.message || "Failed to load tools");
      }
    } catch (error) {
      setTools([]);
      setToolsError(error instanceof Error ? error.message : "Failed to load tools");
    } finally {
      setIsLoadingTools(false);
    }
    return true;
  };

  const fetchTools = async () => {
    if (!accessToken || !mcpServer.server_id) return;

    // OBO/M2M/static auth is attached server-side from the stored credential, so
    // a plain GET /tools/list?server_id suffices. PKCE passthrough holds the token
    // in the browser, so forward it from sessionStorage as the x-mcp header the
    // same way the Tools playground does.
    let customHeaders: Record<string, string> | undefined;
    const isPassthrough =
      getMcpOAuthMode({
        auth_type: mcpServer.auth_type,
        oauth2_flow: mcpServer.oauth2_flow,
        delegate_auth_to_upstream: mcpServer.delegate_auth_to_upstream,
      }) === "passthrough";
    const isBrowserHeldTokenMode = isClientForwardedTokenMode(getEffectiveAuthType());

    if (await previewWithStagedInteractiveToken(isPassthrough, isBrowserHeldTokenMode)) {
      return;
    }
    if (isPassthrough || isBrowserHeldTokenMode) {
      const token =
        oauthTokenResponse?.access_token ??
        (isTokenValid(mcpServer.server_id, userID)
          ? getToken(mcpServer.server_id, userID)?.access_token ?? null
          : null);
      if (!token) {
        setTools([]);
        setToolsError(
          isBrowserHeldTokenMode
            ? "Authorize with the upstream (browser-only, in the Authentication section) to load and configure this server's tools."
            : "Authenticate with this server in the Tools tab to load and configure its tools.",
        );
        return;
      }
      customHeaders = buildMcpPassthroughAuthHeader(mcpServer.alias, token);
    }

    setIsLoadingTools(true);
    setToolsError(null);

    try {
      // include_disabled_tools: configuring the allowlist needs the full server
      // catalog, so tools toggled off still render (as unchecked) instead of vanishing.
      const toolsResponse = await listMCPTools(accessToken, mcpServer.server_id, customHeaders, true);

      if (toolsResponse.tools && !toolsResponse.error) {
        setTools(toolsResponse.tools);
      } else {
        setTools([]);
        setToolsError(toolsResponse.message || "Failed to load tools");
      }
    } catch (error) {
      setTools([]);
      setToolsError(error instanceof Error ? error.message : "Failed to load tools");
    } finally {
      setIsLoadingTools(false);
    }
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

  const handleTransportChange = (value: string) => {
    // Clear fields that are not relevant for the selected transport.
    if (value === "stdio") {
      const clearedForStdio = {
        url: undefined,
        spec_path: undefined,
        auth_type: undefined,
        credentials: undefined,
        issuer: undefined,
        authorization_url: undefined,
        token_url: undefined,
        registration_url: undefined,
      };
      form.setFieldsValue(clearedForStdio);
    } else if (value === TRANSPORT.OPENAPI) {
      const clearedForOpenapi = {
        url: undefined,
        command: undefined,
        args: undefined,
        env_json: undefined,
        stdio_config: undefined,
      };
      form.setFieldsValue(clearedForOpenapi);
    } else {
      form.setFieldsValue({
        spec_path: undefined,
        command: undefined,
        args: undefined,
        env_json: undefined,
        stdio_config: undefined,
      });
    }
    if (isHeldOAuthTokenStale(form.getFieldsValue(true), authorizedIdentityRef.current)) {
      clearHeldOAuthToken();
    }
  };

  const handleSave = async (values: Record<string, any>) => {
    if (!accessToken) return;
    const invalidDisplayName = Object.entries(toolNameToDisplayName).find(
      ([, displayName]) => displayName && !TOOL_DISPLAY_NAME_PATTERN.test(displayName),
    );
    if (invalidDisplayName) {
      NotificationsManager.fromBackend(
        `Tool display name "${invalidDisplayName[1]}" is invalid. Only letters, digits, underscores, and hyphens are allowed (no spaces).`,
      );
      return;
    }
    try {
      // Ensure access groups is always a string array
      const {
        static_headers: staticHeadersList,
        env_vars: envVarsList,
        credentials: credentialValues,
        stdio_config: rawStdioConfig,
        env_json: rawEnvJson,
        command: rawCommand,
        args: rawArgs,
        allow_all_keys: allowAllKeysRaw,
        available_on_public_internet: availableOnPublicInternetRaw,
        delegate_auth_to_upstream: delegateAuthToUpstreamRaw,
        oauth_passthrough: oauthPassthroughRaw,
        dcr_bridge: dcrBridgeRaw,
        token_validation_json: rawTokenValidationJson,
        ...restValues
      } = values;

      const accessGroups = (restValues.mcp_access_groups || []).map((g: any) =>
        typeof g === "string" ? g : g.name || String(g),
      );

      const staticHeaders = Array.isArray(staticHeadersList)
        ? staticHeadersList.reduce((acc: Record<string, string>, entry: Record<string, string>) => {
            const header = entry?.header?.trim();
            if (!header) {
              return acc;
            }
            acc[header] = (entry?.value ?? "").trim();
            return acc;
          }, {})
        : ({} as Record<string, string>);

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

      let stdioFields: Record<string, any> = {};

      if (restValues.transport === "stdio") {
        // Prefer JSON config if provided (matches Create screen behavior)
        if (rawStdioConfig) {
          try {
            const stdioConfig = JSON.parse(rawStdioConfig);

            let actualConfig = stdioConfig;
            if (stdioConfig?.mcpServers && typeof stdioConfig.mcpServers === "object") {
              const serverNames = Object.keys(stdioConfig.mcpServers);
              if (serverNames.length > 0) {
                actualConfig = stdioConfig.mcpServers[serverNames[0]];
              }
            }

            const parsedArgs = Array.isArray(actualConfig?.args)
              ? actualConfig.args.map((v: any) => String(v)).filter((v: string) => v.trim() !== "")
              : [];

            const parsedEnv =
              actualConfig?.env && typeof actualConfig.env === "object" && !Array.isArray(actualConfig.env)
                ? Object.entries(actualConfig.env).reduce((acc: Record<string, string>, [k, v]) => {
                    if (k == null || String(k).trim() === "") return acc;
                    acc[String(k)] = v == null ? "" : String(v);
                    return acc;
                  }, {})
                : {};

            stdioFields = {
              command: actualConfig?.command ? String(actualConfig.command) : undefined,
              args: parsedArgs,
              env: parsedEnv,
            };

            if (!stdioFields.command) {
              NotificationsManager.fromBackend("Stdio configuration must include a command");
              return;
            }
          } catch {
            NotificationsManager.fromBackend("Invalid JSON in stdio configuration");
            return;
          }
        } else {
          // Dedicated fields path (command/args + env JSON)
          let parsedEnv: Record<string, string> = {};
          if (rawEnvJson) {
            try {
              const env = JSON.parse(rawEnvJson);
              if (env && typeof env === "object" && !Array.isArray(env)) {
                parsedEnv = Object.entries(env).reduce((acc: Record<string, string>, [k, v]) => {
                  if (k == null || String(k).trim() === "") return acc;
                  acc[String(k)] = v == null ? "" : String(v);
                  return acc;
                }, {});
              }
            } catch {
              NotificationsManager.fromBackend("Invalid JSON in stdio env configuration");
              return;
            }
          }
          const parsedArgs = Array.isArray(rawArgs)
            ? rawArgs.map((v: any) => String(v)).filter((v: string) => v.trim() !== "")
            : [];

          const parsedCommand = rawCommand ? String(rawCommand).trim() : "";
          if (!parsedCommand) {
            NotificationsManager.fromBackend("Stdio transport requires a command");
            return;
          }

          stdioFields = {
            command: parsedCommand,
            args: parsedArgs,
            env: parsedEnv,
          };
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
          return;
        }
      }

      // Prepare the payload with cost configuration and permission fields
      const mcpInfoServerName =
        restValues.server_name ||
        restValues.url ||
        mcpServer.server_name ||
        mcpServer.url ||
        restValues.alias ||
        mcpServer.alias ||
        "unknown";

      const toolAllowlistEnforced = hasExistingToolAllowlist || hasToolAllowlistInteraction || allowedTools.length > 0;

      const payload: Record<string, any> = {
        ...restValues,
        ...stdioFields,
        // Remove UI-only fields
        stdio_config: undefined,
        env_json: undefined,
        ...(mcpServer.auth_type === AUTH_TYPE.OAUTH2 && restValues.auth_type !== AUTH_TYPE.OAUTH2
          ? { issuer: null, authorization_url: null, token_url: null, registration_url: null }
          : {}),
        ...(mcpServer.auth_type === AUTH_TYPE.OAUTH2_TOKEN_EXCHANGE &&
        restValues.auth_type !== AUTH_TYPE.OAUTH2_TOKEN_EXCHANGE
          ? { token_exchange_endpoint: null, audience: null, subject_token_type: null, token_exchange_profile: null }
          : {}),
        server_id: mcpServer.server_id,
        mcp_info: {
          ...(mcpServer.mcp_info ?? {}),
          server_name: mcpInfoServerName,
          description: restValues.description,
          logo_url: logoUrl || undefined,
          mcp_server_cost_info: Object.keys(costConfig).length > 0 ? costConfig : null,
          tool_allowlist_enforced: toolAllowlistEnforced,
        },
        mcp_access_groups: accessGroups,
        alias: restValues.alias,
        // Include permission management fields
        extra_headers: restValues.extra_headers || [],
        ...(toolAllowlistEnforced
          ? {
              allowed_tools: allowedTools,
            }
          : {}),
        tool_name_to_display_name: Object.keys(toolNameToDisplayName).length > 0 ? toolNameToDisplayName : null,
        tool_name_to_description: Object.keys(toolNameToDescription).length > 0 ? toolNameToDescription : null,
        disallowed_tools: restValues.disallowed_tools || [],
        static_headers: staticHeaders,
        env_vars: envVars,
        allow_all_keys: Boolean(allowAllKeysRaw ?? mcpServer.allow_all_keys),
        available_on_public_internet: Boolean(availableOnPublicInternetRaw ?? mcpServer.available_on_public_internet),
        // ``delegate_auth_to_upstream`` is only honored server-side for
        // ``auth_type=oauth2`` (PKCE passthrough). The Form.Item is
        // conditionally rendered so the value drops out of the form on
        // auth_type change; force false for any other configuration to avoid
        // persisting a stale ``true`` that would silently re-activate if the
        // configuration is later switched back.
        delegate_auth_to_upstream: (() => {
          const isOauth2 = restValues.auth_type === AUTH_TYPE.OAUTH2;
          return isOauth2 ? Boolean(delegateAuthToUpstreamRaw ?? mcpServer.delegate_auth_to_upstream) : false;
        })(),
        // ``oauth_passthrough`` is the dedicated, non-oauth2 opt-in. It is only
        // honored for ``auth_type=none`` servers that forward ``Authorization``
        // upstream. Kept separate from ``delegate_auth_to_upstream`` so enabling
        // pass-through never regresses oauth2 servers. Force false otherwise.
        oauth_passthrough: (() => {
          const isNoneAuth = restValues.auth_type === AUTH_TYPE.NONE || restValues.auth_type == null;
          const extraHeaders = Array.isArray(restValues.extra_headers) ? restValues.extra_headers : [];
          const hasAuthorizationHeader = extraHeaders.some(
            (h: unknown) => typeof h === "string" && h.toLowerCase() === "authorization",
          );
          return isNoneAuth && hasAuthorizationHeader
            ? Boolean(oauthPassthroughRaw ?? mcpServer.oauth_passthrough)
            : false;
        })(),
        // ``dcr_bridge`` is only meaningful for the client-forwarded token
        // modes (true_passthrough / oauth_delegate). The Form.Item is
        // conditionally rendered so the value drops out of the form on
        // auth_type change; force false for any other configuration to avoid
        // persisting a stale ``true`` that would silently re-activate if the
        // mode is later switched back.
        dcr_bridge: isClientForwardedTokenMode(restValues.auth_type)
          ? Boolean(dcrBridgeRaw ?? mcpServer.dcr_bridge)
          : false,
        ...(restValues.auth_type === AUTH_TYPE.OAUTH2 && restValues.oauth_flow_type
          ? {
              oauth2_flow:
                restValues.oauth_flow_type === OAUTH_FLOW.M2M ? MCP_OAUTH2_FLOW_M2M : MCP_OAUTH2_FLOW_INTERACTIVE,
            }
          : {}),
        // Include token_validation when it is set (non-null) or when clearing an existing value
        ...(tokenValidation !== null || mcpServer.token_validation ? { token_validation: tokenValidation } : {}),
      };

      const includeCredentials =
        restValues.auth_type && AUTH_TYPES_REQUIRING_CREDENTIALS.includes(restValues.auth_type);

      // Client-forwarded rows persist ONLY the declared app; strip any token material lingering in the
      // form (e.g. from a prior oauth2 authorize this session) so it can never reach the row.
      const submitCredentials = isClientForwardedTokenMode(restValues.auth_type)
        ? preservedDeclaredAppCredentials(credentialsPayload)
        : credentialsPayload;

      if (includeCredentials && submitCredentials && Object.keys(submitCredentials).length > 0) {
        payload.credentials = submitCredentials;
      }

      // Explicit removal of a saved app for the client-forwarded modes, applied AFTER the filter so it
      // always wins. Blank fields are the keep-existing convention (the backend merges partial
      // credential updates), so removal must be an explicit-null write: encrypt skips nulls and the
      // merge overrides the stored keys, returning the server to dynamic client registration.
      if (removeStoredApp && isClientForwardedTokenMode(restValues.auth_type)) {
        payload.credentials = { client_id: null, client_secret: null };
      }

      const updated = await updateMCPServer(accessToken, payload);

      // Persist the token staged via "Authorize & Fetch" (mirrors the create flow's
      // commit-on-submit): OBO writes the per-user token to the DB; legacy passthrough and the
      // client-forwarded modes (true_passthrough / oauth_delegate) keep it in sessionStorage and
      // never in the server row. M2M/static auth resolve server-side and need neither.
      if (oauthTokenResponse?.access_token) {
        const oauthMode = getMcpOAuthMode({
          auth_type: restValues.auth_type,
          oauth2_flow: isM2MFlow ? MCP_OAUTH2_FLOW_M2M : null,
          delegate_auth_to_upstream: Boolean(delegateAuthToUpstreamRaw ?? mcpServer.delegate_auth_to_upstream),
        });
        try {
          if (oauthMode === "authorization_code") {
            const scope = oauthTokenResponse.scope;
            const oauthCredentialPayload = {
              access_token: oauthTokenResponse.access_token,
              refresh_token: oauthTokenResponse.refresh_token,
              expires_in: oauthTokenResponse.expires_in,
              scopes: typeof scope === "string" && scope ? scope.split(" ") : undefined,
            };
            await storeMCPOAuthUserCredential(accessToken, mcpServer.server_id, oauthCredentialPayload);
          } else if (oauthMode === "passthrough" || isClientForwardedTokenMode(restValues.auth_type)) {
            const browserHeldToken = {
              access_token: oauthTokenResponse.access_token,
              expires_in: oauthTokenResponse.expires_in,
              refresh_token: oauthTokenResponse.refresh_token,
              token_type: oauthTokenResponse.token_type,
            };
            setToken(mcpServer.server_id, browserHeldToken, userID);
          }
        } catch (error: unknown) {
          const message = error instanceof Error ? error.message : "";
          NotificationsManager.fromBackend(
            "MCP Server updated, but failed to persist OAuth token" + (message ? `: ${message}` : ""),
          );
          return;
        }
      }

      NotificationsManager.success("MCP Server updated successfully");
      setAppMayNotMatchUpstream(false);
      onSuccess(updated);
    } catch (error: any) {
      NotificationsManager.fromBackend("Failed to update MCP Server" + (error?.message ? `: ${error.message}` : ""));
    }
  };

  return (
    <TabGroup>
      <TabList className="grid w-full grid-cols-2">
        <Tab>Server Configuration</Tab>
        <Tab>Cost Configuration</Tab>
      </TabList>
      <TabPanels className="mt-6">
        <TabPanel>
          <Form
            form={form}
            onFinish={handleSave}
            onValuesChange={handleFormValuesChange}
            initialValues={initialValues}
            layout="vertical"
          >
            <Form.Item
              label="MCP Server Name"
              name="server_name"
              rules={[
                {
                  validator: (_, value) => validateMCPServerName(value),
                },
              ]}
            >
              <Input className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500" />
            </Form.Item>
            <Form.Item
              label="Alias"
              name="alias"
              rules={[
                {
                  validator: (_, value) => validateMCPServerName(value),
                },
              ]}
            >
              <Input
                onChange={() => setAliasManuallyEdited(true)}
                className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
              />
            </Form.Item>
            <Form.Item label="Description" name="description">
              <Input className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500" />
            </Form.Item>
            <MCPLogoSelector value={logoUrl} onChange={setLogoUrl} />
            <Form.Item label="Transport Type" name="transport" rules={[{ required: true }]}>
              <Select onChange={handleTransportChange}>
                <Select.Option value="http">Streamable HTTP (Recommended)</Select.Option>
                <Select.Option value="sse">Server-Sent Events (SSE)</Select.Option>
                <Select.Option value="stdio">Standard Input/Output (stdio)</Select.Option>
                <Select.Option value={TRANSPORT.OPENAPI}>OpenAPI Spec</Select.Option>
              </Select>
            </Form.Item>

            {/* URL field - only for HTTP/SSE */}
            {isMCPTransport && (
              <Form.Item
                label="MCP Server URL"
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

            {/* OpenAPI Spec URL - only for OpenAPI transport */}
            {isOpenAPITransport && (
              <Form.Item
                label={
                  <span className="text-sm font-medium text-gray-700 flex items-center">
                    OpenAPI Spec URL
                    <Tooltip title="URL to an OpenAPI specification (JSON or YAML). MCP tools will be automatically generated from the API endpoints defined in the spec.">
                      <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                    </Tooltip>
                  </span>
                }
                name="spec_path"
                rules={[{ required: true, message: "Please enter an OpenAPI spec URL" }]}
              >
                <Input
                  placeholder="https://petstore3.swagger.io/api/v3/openapi.json"
                  className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                />
              </Form.Item>
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

            {/* Authentication - for HTTP, SSE, and OpenAPI */}
            {!isStdioTransport && (
              <>
                <Form.Item label="Authentication" name="auth_type" rules={[{ required: true }]}>
                  <Select>
                    <Select.Option value="none">None</Select.Option>
                    <Select.Option value="api_key">API Key</Select.Option>
                    <Select.Option value="bearer_token">Bearer Token</Select.Option>
                    <Select.Option value="token">Token</Select.Option>
                    <Select.Option value="basic">Basic Auth</Select.Option>
                    <Select.Option value="oauth2">OAuth</Select.Option>
                    <Select.Option value="oauth2_token_exchange">OAuth Token Exchange (OBO)</Select.Option>
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
                  oauthFlow={{
                    startOAuthFlow,
                    status: oauthStatus,
                    error: oauthError,
                    tokenResponse: oauthTokenResponse,
                  }}
                  isEditing
                  savedAuthType={mcpServer.auth_type}
                  removeStoredApp={removeStoredApp}
                  onRemoveStoredAppChange={setRemoveStoredApp}
                  appMayNotMatchUpstream={appMayNotMatchUpstream}
                />
              </>
            )}

            {isStdioTransport && (
              <div className="rounded-lg border border-gray-200 p-4 space-y-4">
                <p className="text-sm text-gray-600">
                  Configure the stdio transport used to launch the MCP server process. You can either fill in the fields
                  below or paste a JSON configuration.
                </p>

                <Form.Item
                  label="Command"
                  name="command"
                  rules={[{ required: true, message: "Please enter a command for stdio transport" }]}
                >
                  <Input
                    placeholder="e.g., npx"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>

                <Form.Item label="Args" name="args">
                  <Select
                    mode="tags"
                    size="large"
                    tokenSeparators={[","]}
                    placeholder="Add args (press enter or comma)"
                    className="rounded-lg"
                  />
                </Form.Item>

                <Form.Item
                  label="Environment (JSON object)"
                  name="env_json"
                  rules={[
                    {
                      validator: (_, value) => {
                        if (!value) return Promise.resolve();
                        try {
                          const parsed = JSON.parse(value);
                          if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
                            return Promise.resolve();
                          }
                          return Promise.reject(new Error("Env must be a JSON object"));
                        } catch {
                          return Promise.reject(new Error("Please enter valid JSON"));
                        }
                      },
                    },
                  ]}
                >
                  <Input.TextArea
                    rows={6}
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500 font-mono text-sm"
                    placeholder={`{\n  \"KEY\": \"value\"\n}`}
                  />
                </Form.Item>

                {/* Optional JSON config (if provided, it overrides command/args/env on save) */}
                <StdioConfiguration isVisible={true} required={false} />
              </div>
            )}

            {!isStdioTransport && shouldShowAuthValueField && (
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
                        ? Promise.reject(new Error("Authentication value cannot be empty"))
                        : Promise.resolve(),
                  },
                ]}
              >
                <Input.Password
                  placeholder="Enter token or secret (leave blank to keep existing)"
                  className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                />
              </Form.Item>
            )}

            {!isStdioTransport && isOAuthAuthType && (
              <>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      OAuth Flow Type
                      <Tooltip title="Machine-to-Machine (M2M) authenticates with client credentials and no user interaction. Interactive (PKCE) authorizes each user in the browser and stores per-user tokens. Servers created before this field existed have no stored value; choose one to persist it.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name="oauth_flow_type"
                >
                  <Select placeholder="Select OAuth flow">
                    <Select.Option value={OAUTH_FLOW.M2M}>Machine-to-Machine (M2M)</Select.Option>
                    <Select.Option value={OAUTH_FLOW.INTERACTIVE}>Interactive (PKCE)</Select.Option>
                  </Select>
                </Form.Item>
                {!oauthFlowTypeValue && !isDelegateAuth && (
                  <Alert
                    type="warning"
                    showIcon
                    className="mb-4 rounded-lg"
                    message="This server has no OAuth flow set"
                    description="Choose Machine-to-Machine (M2M) or Interactive (PKCE) so LiteLLM authenticates it the way you intend, then save. Until it is set, LiteLLM falls back to interactive per-user auth and treats a machine-to-machine credential shape conservatively."
                  />
                )}
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      OAuth Client ID (optional)
                      <Tooltip title="Provide only if your MCP server cannot handle dynamic client registration.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name={["credentials", "client_id"]}
                >
                  <Input.Password
                    placeholder="Enter OAuth client ID (leave blank to keep existing)"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      OAuth Client Secret (optional)
                      <Tooltip title="Provide only if your MCP server cannot handle dynamic client registration.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name={["credentials", "client_secret"]}
                >
                  <Input.Password
                    placeholder="Enter OAuth client secret (leave blank to keep existing)"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      OAuth Scopes (optional)
                      <Tooltip title="Add scopes to override the default scope list used for this MCP server.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name={["credentials", "scopes"]}
                >
                  <Select
                    mode="tags"
                    tokenSeparators={[","]}
                    placeholder="Add scopes"
                    className="rounded-lg"
                    size="large"
                  />
                </Form.Item>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      Issuer (optional)
                      <Tooltip title="OAuth 2.0 authorization server issuer (RFC 8414). Auto-discovered on first connect; set it explicitly to pin the trust anchor so token and scope discovery is fetched from and validated against this issuer (RFC 8414 §3.3) instead of anything the resource advertises.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name="issuer"
                >
                  <Input
                    placeholder="https://issuer.example.com"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      Authorization URL Override (optional)
                      <Tooltip title="Optional override for the authorization endpoint.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name="authorization_url"
                >
                  <Input
                    placeholder="https://example.com/oauth/authorize"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      Token URL Override (optional)
                      <Tooltip title="Optional override for the token endpoint.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name="token_url"
                >
                  <Input
                    placeholder="https://example.com/oauth/token"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
                <TokenEndpointAuthMethodField isEditing />
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      Registration URL Override (optional)
                      <Tooltip title="Optional override for the dynamic client registration endpoint.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name="registration_url"
                >
                  <Input
                    placeholder="https://example.com/oauth/register"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
                {!isM2MFlow && (
                  <>
                    <Form.Item
                      label={
                        <span className="text-sm font-medium text-gray-700 flex items-center">
                          Token Validation Rules (optional)
                          <Tooltip title='JSON object of key-value rules checked against the OAuth token response before storing. Supports dot-notation for nested fields (e.g. {"organization": "my-org", "team.id": "123"}). Tokens that fail validation are rejected with HTTP 403.'>
                            <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                          </Tooltip>
                        </span>
                      }
                      name="token_validation_json"
                      rules={[
                        {
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
                      <Input.TextArea
                        placeholder={'{\n  "organization": "my-org",\n  "team.id": "123"\n}'}
                        rows={4}
                        className="font-mono text-sm rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                      />
                    </Form.Item>
                    <Form.Item
                      label={
                        <span className="text-sm font-medium text-gray-700 flex items-center">
                          Token Storage TTL (seconds, optional)
                          <Tooltip title="How long to cache each user's OAuth access token in Redis before evicting it (never longer than the token's own expires_in). Leave blank to derive the TTL from the token's expires_in, or fall back to the 12-hour default.">
                            <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                          </Tooltip>
                        </span>
                      }
                      name="token_storage_ttl_seconds"
                    >
                      <InputNumber min={1} placeholder="e.g. 3600" style={{ width: "100%" }} className="rounded-lg" />
                    </Form.Item>
                  </>
                )}
                <div className="rounded-lg border border-dashed border-gray-300 p-4 space-y-2">
                  <p className="text-sm text-gray-600">
                    Use OAuth to fetch a fresh access token and temporarily save it in the session as the authentication
                    value.
                  </p>
                  <Button
                    variant="secondary"
                    onClick={startOAuthFlow}
                    disabled={oauthStatus === "authorizing" || oauthStatus === "exchanging"}
                  >
                    {oauthStatus === "authorizing"
                      ? "Waiting for authorization..."
                      : oauthStatus === "exchanging"
                        ? "Exchanging authorization code..."
                        : "Authorize & Fetch Token"}
                  </Button>
                  {oauthError && <p className="text-sm text-red-500">{oauthError}</p>}
                  {oauthStatus === "success" && oauthTokenResponse?.access_token && (
                    <p className="text-sm text-green-600">
                      Token fetched. Expires in {oauthTokenResponse.expires_in ?? "?"} seconds.
                    </p>
                  )}
                </div>
              </>
            )}

            {!isStdioTransport && isTokenExchangeAuthType && <TokenExchangeFormFields isEditing />}

            {!isStdioTransport && isAwsSigV4AuthType && (
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
                  rules={[]}
                >
                  <Input
                    placeholder="us-east-1 (leave blank to keep existing)"
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
                    placeholder="bedrock-agentcore (leave blank to keep existing)"
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
                  rules={[]}
                >
                  <Input.Password
                    placeholder="Leave blank to keep existing"
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
                  rules={[]}
                >
                  <Input.Password
                    placeholder="Leave blank to keep existing"
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
                    placeholder="Leave blank to keep existing"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
                <Form.Item
                  label={
                    <span className="text-sm font-medium text-gray-700 flex items-center">
                      AWS Role ARN
                      <Tooltip title="Optional. IAM role ARN to assume via STS before signing. If set, LiteLLM calls sts:AssumeRole to get temporary credentials.">
                        <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
                      </Tooltip>
                    </span>
                  }
                  name={["credentials", "aws_role_name"]}
                >
                  <Input
                    placeholder="Leave blank to keep existing"
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
                    placeholder="Leave blank to keep existing"
                    className="rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                  />
                </Form.Item>
              </>
            )}

            {/* Environment Variables Section */}
            <div className="mt-6">
              <EnvVarsSection />
            </div>

            {/* Permission Management / Access Control Section */}
            <div className="mt-6">
              <MCPPermissionManagement
                availableAccessGroups={availableAccessGroups}
                mcpServer={mcpServer}
                searchValue={searchValue}
                setSearchValue={setSearchValue}
                getAccessGroupOptions={getAccessGroupOptions}
              />
            </div>

            {/* Tool Configuration Section */}
            <div className="mt-6">
              <MCPToolConfiguration
                accessToken={accessToken}
                formValues={{
                  server_id: mcpServer.server_id,
                  server_name: currentServerName ?? mcpServer.server_name,
                  url: currentUrl ?? mcpServer.url,
                  spec_path: currentSpecPath ?? mcpServer.spec_path,
                  transport: transportType ?? mcpServer.transport,
                  auth_type: currentAuthType ?? mcpServer.auth_type,
                  mcp_info: mcpServer.mcp_info,
                  oauth_flow_type:
                    oauthFlowTypeValue ?? oauth2FlowToFormValue(mcpServer.oauth2_flow) ?? OAUTH_FLOW.INTERACTIVE,
                  static_headers: currentStaticHeaders ?? mcpServer.static_headers,
                  credentials: currentCredentials,
                  issuer: currentIssuer ?? mcpServer.issuer,
                  authorization_url: currentAuthorizationUrl ?? mcpServer.authorization_url,
                  token_url: currentTokenUrl ?? mcpServer.token_url,
                  registration_url: currentRegistrationUrl ?? mcpServer.registration_url,
                }}
                allowedTools={allowedTools}
                existingAllowedTools={existingAllowedTools}
                hasToolAllowlistInteraction={hasToolAllowlistInteraction}
                isEditMode
                onAllowedToolsChange={setAllowedTools}
                onToolAllowlistInteraction={() => setHasToolAllowlistInteraction(true)}
                toolNameToDisplayName={toolNameToDisplayName}
                toolNameToDescription={toolNameToDescription}
                onToolNameToDisplayNameChange={setToolNameToDisplayName}
                onToolNameToDescriptionChange={setToolNameToDescription}
                externalTools={tools}
                externalIsLoading={isLoadingTools}
                externalError={toolsError}
                externalCanFetch={true}
              />
            </div>

            <div className="flex justify-end gap-2">
              <AntdButton onClick={onCancel}>Cancel</AntdButton>
              <Button type="submit">Save Changes</Button>
            </div>
          </Form>
        </TabPanel>

        <TabPanel>
          <div className="space-y-6">
            <MCPServerCostConfig value={costConfig} onChange={setCostConfig} tools={tools} disabled={isLoadingTools} />

            <div className="flex justify-end gap-2">
              <AntdButton onClick={onCancel}>Cancel</AntdButton>
              <Button onClick={() => form.submit()}>Save Changes</Button>
            </div>
          </div>
        </TabPanel>
      </TabPanels>
    </TabGroup>
  );
};

export default MCPServerEdit;
