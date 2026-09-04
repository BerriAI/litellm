import React, { useState, useEffect } from "react";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { Info, TriangleAlert } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { FormProvider, useForm } from "react-hook-form";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  AUTH_TYPE,
  isClientForwardedTokenMode,
  getOAuthAuthorizationIdentity,
  CLEARED_ON_INVALIDATION,
  isHeldOAuthTokenStale,
  preservedAdminCredentials,
  preservedDeclaredAppCredentials,
  withoutMintedTokenCredentials,
  OAUTH_FLOW,
  MCP_OAUTH2_FLOW_M2M,
  MCP_OAUTH2_FLOW_INTERACTIVE,
  MCPServer,
  MCPServerCostInfo,
  TRANSPORT,
  TRANSPORT_ITEMS,
  AUTH_TYPE_ITEMS,
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
import IdJagFormFields from "./IdJagFormFields";
import OAuthFormFields from "./OAuthFormFields";
import MCPLogoSelector from "./MCPLogoSelector";
import EnvVarsSection from "./EnvVarsSection";
import { validateMCPServerUrl, validateMCPServerName, normalizeToolOverrideMap } from "./utils";
import { EditServerFormValues, buildEditServerPayload, editPayloadErrorMessage } from "./editServerPayload";
import { toast } from "@/lib/toast";
import { useMcpOAuthFlow } from "@/hooks/useMcpOAuthFlow";
import {
  MountedFormField,
  MountedFormProvider,
  projectMountedValues,
  useMountRegistry,
  type MountedFormValues,
} from "@/components/common_components/MountedFormField";
import { requiredRule, validatorRules } from "@/components/common_components/formRules";
import {
  allFieldsValue,
  mountedPaths,
  resetFields,
  setFieldsValue,
  singleBranchChange,
  useMountedValues,
} from "./mcpFormStore";
import {
  numberControl,
  notOnlyWhitespace,
  parsesAsJsonObject,
  selectControl,
  selectTriggerControl,
  tagsControl,
  textControl,
} from "./mcpFieldRules";
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
export const EDIT_OAUTH_UI_STATE_KEY = "litellm-mcp-oauth-edit-state";

const MCPServerEdit: React.FC<MCPServerEditProps> = ({
  mcpServer,
  accessToken,
  userID,
  onCancel,
  onSuccess,
  availableAccessGroups,
}) => {
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

  const form = useForm<MountedFormValues>({ mode: "onChange", defaultValues: initialValues });
  const registry = useMountRegistry();
  const mountedValues = useMountedValues(form, registry);
  const [costConfig, setCostConfig] = useState<MCPServerCostInfo>({});
  const [tools, setTools] = useState<any[]>([]);
  const [isLoadingTools, setIsLoadingTools] = useState(false);
  const [toolsError, setToolsError] = useState<string | null>(null);
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
  const authType = mountedValues.auth_type as string | undefined;
  const transportType = mountedValues.transport as string | undefined;
  const isStdioTransport = transportType === "stdio";
  const isOpenAPITransport = transportType === TRANSPORT.OPENAPI;
  const isMCPTransport = !isStdioTransport && !isOpenAPITransport;
  const shouldShowAuthValueField = authType ? AUTH_TYPES_REQUIRING_AUTH_VALUE.includes(authType) : false;
  const isOAuthAuthType = authType === AUTH_TYPE.OAUTH2;
  const isTokenExchangeAuthType = authType === AUTH_TYPE.OAUTH2_TOKEN_EXCHANGE;
  const isIdJagAuthType = authType === AUTH_TYPE.OAUTH2_ID_JAG;
  const isAwsSigV4AuthType = authType === AUTH_TYPE.AWS_SIGV4;
  // Same fallback as the delegate switch below: the value is undefined until the field mounts, so
  // reading it alone flashes the "no OAuth flow set" warning at a server that already has one.
  const oauthFlowTypeValue =
    (mountedValues.oauth_flow_type as string | undefined) ?? oauth2FlowToFormValue(mcpServer.oauth2_flow);
  const isM2MFlow = isOAuthAuthType && oauthFlowTypeValue === OAUTH_FLOW.M2M;
  // Watch reflects a live toggle when the delegate switch is mounted; fall back to
  // the stored value otherwise (useWatch returns undefined for an unmounted field,
  // the same trap the oauth_flow_type field originally hit).
  const delegateAuthWatched = mountedValues.delegate_auth_to_upstream as boolean | undefined;
  const isDelegateAuth = delegateAuthWatched ?? Boolean(mcpServer.delegate_auth_to_upstream);

  // Watch form fields that affect tool fetching
  const currentUrl = mountedValues.url;
  const currentSpecPath = mountedValues.spec_path;
  const currentServerName = mountedValues.server_name;
  const currentAuthType = mountedValues.auth_type;
  const currentStaticHeaders = mountedValues.static_headers;
  const currentCredentials = mountedValues.credentials;
  const currentIssuer = mountedValues.issuer;
  const currentAuthorizationUrl = mountedValues.authorization_url;
  const currentTokenUrl = mountedValues.token_url;
  const currentRegistrationUrl = mountedValues.registration_url;
  const hasExistingToolAllowlist =
    Boolean(mcpServer.mcp_info?.tool_allowlist_enforced) || (mcpServer.allowed_tools?.length ?? 0) > 0;
  const existingAllowedTools = hasExistingToolAllowlist ? mcpServer.allowed_tools ?? [] : null;

  const persistEditUiState = () => {
    if (typeof window === "undefined") {
      return;
    }
    try {
      const values = allFieldsValue(form);
      setSecureItem(
        EDIT_OAUTH_UI_STATE_KEY,
        JSON.stringify({
          serverId: mcpServer.server_id,
          formValues: values,
          costConfig,
          allowedTools,
          hasToolAllowlistInteraction,
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
  const getEffectiveAuthType = () => allFieldsValue(form).auth_type ?? mcpServer.auth_type;

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
    getCredentials: () => allFieldsValue(form).credentials,
    getTemporaryPayload: () => {
      const values = allFieldsValue(form);
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
          ? preservedAdminCredentials(values.credentials)
          : values.credentials,
        issuer: values.issuer,
        authorization_url: values.authorization_url,
        token_url: values.token_url,
        registration_url: values.registration_url,
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

      authorizedIdentityRef.current = getOAuthAuthorizationIdentity(allFieldsValue(form));
      if (isClientForwardedTokenMode(getEffectiveAuthType())) {
        const browserHeldToken = {
          access_token: token.access_token,
          expires_in: token.expires_in,
          token_type: token.token_type,
        };
        setToken(mcpServer.server_id, browserHeldToken, userID);
        toast.success(
          "Token held for this browser session. Tools can now be loaded and configured; the token is not saved to LiteLLM.",
        );
        return;
      }

      const current = (allFieldsValue(form).credentials as Record<string, unknown> | undefined) ?? {};
      const nextCredentials = {
        ...(preservedAdminCredentials(current) ?? {}),
        ...(current.scopes !== undefined && { scopes: current.scopes }),
        access_token: token.access_token,
        ...(token.refresh_token && { refresh_token: token.refresh_token }),
        ...(token.expires_in && { expires_in: token.expires_in }),
        ...(token.scope && { scope: token.scope }),
      };
      // Path-replace (not deep-merge) so a re-authorize with fewer token fields does not leave stale
      // siblings behind; the admin-typed client keys and scopes are carried explicitly above.
      form.setValue("credentials", nextCredentials);
      // Re-capture after writing credentials so the token is not invalidated by its own credential write.
      authorizedIdentityRef.current = getOAuthAuthorizationIdentity(allFieldsValue(form));

      toast.success("OAuth authorization successful! Please click 'Update MCP Server' to save the credentials.");
    },
    onBeforeRedirect: persistEditUiState,
    flowSource: "edit",
  });

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
    setFieldsValue(form, initialValues);
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
    if (transport && transport !== allFieldsValue(form).transport) {
      setFieldsValue(form, { transport });
      return;
    }
    setFieldsValue(form, pendingRestoredValues);
    setPendingRestoredValues(null);
  }, [pendingRestoredValues, form, mcpServer.transport, transportType]);

  // Transform string array to object array for initial form values
  useEffect(() => {
    if (mcpServer.mcp_access_groups) {
      // If access groups are objects, extract the name property; if strings, use as is
      const groupNames = mcpServer.mcp_access_groups.map((g: any) => (typeof g === "string" ? g : g.name || String(g)));
      form.setValue("mcp_access_groups", groupNames);
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
    const keptAdminCredentials = preservedAdminCredentials(allFieldsValue(form).credentials);
    resetFields(form, [...CLEARED_ON_INVALIDATION], initialValues as MountedFormValues);
    if (keptAdminCredentials) {
      setFieldsValue(form, { credentials: keptAdminCredentials });
    }
    const preserved = Object.fromEntries(
      CLEARED_ON_INVALIDATION.filter((key) => key in changedValues).map((key) => [key, changedValues[key]]),
    );
    if (Object.keys(preserved).length > 0) {
      setFieldsValue(form, preserved);
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
      const hasDeclaredApp = preservedDeclaredAppCredentials(allFieldsValue(form).credentials) !== undefined;
      if (upstreamChanged && hasDeclaredApp) {
        setAppMayNotMatchUpstream(true);
      }
    }
    if (isHeldOAuthTokenStale(allFieldsValue(form), authorizedIdentityRef.current)) {
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
      const values = allFieldsValue(form);
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
      setFieldsValue(form, clearedForStdio);
    } else if (value === TRANSPORT.OPENAPI) {
      const clearedForOpenapi = {
        url: undefined,
        command: undefined,
        args: undefined,
        env_json: undefined,
        stdio_config: undefined,
      };
      setFieldsValue(form, clearedForOpenapi);
    } else {
      setFieldsValue(form, {
        spec_path: undefined,
        command: undefined,
        args: undefined,
        env_json: undefined,
        stdio_config: undefined,
      });
    }
    if (isHeldOAuthTokenStale(allFieldsValue(form), authorizedIdentityRef.current)) {
      clearHeldOAuthToken();
    }
  };

  const handleTransportSelected =
    (onChange: (value: string) => void) =>
    (value: string | null): void => {
      if (value === null) return;
      onChange(value);
      handleTransportChange(value);
    };

  const valuesChangeRef = React.useRef(handleFormValuesChange);
  valuesChangeRef.current = handleFormValuesChange;

  React.useEffect(() => {
    const subscription = form.watch((values, { name, type }) => {
      if (type !== "change" || name === undefined) {
        return;
      }
      valuesChangeRef.current(singleBranchChange(name, values as MountedFormValues));
    });
    return () => subscription.unsubscribe();
  }, [form]);

  const submitForm = async () => {
    const isValid = await form.trigger(mountedPaths(registry) as string[]);
    if (!isValid) {
      return;
    }
    await handleSave(projectMountedValues(registry, form.getValues) as unknown as EditServerFormValues);
  };

  const handleSave = async (values: EditServerFormValues) => {
    if (!accessToken) return;
    try {
      const built = buildEditServerPayload(values, {
        mcpServer,
        logoUrl,
        costConfig,
        allowedTools,
        hasExistingToolAllowlist,
        hasToolAllowlistInteraction,
        toolNameToDisplayName,
        toolNameToDescription,
        removeStoredApp,
      });
      if (built.kind !== "ok") {
        toast.fromError(editPayloadErrorMessage(built));
        return;
      }
      const payload = built.payload;

      const updated = await updateMCPServer(accessToken, payload);

      // Persist the token staged via "Authorize & Fetch" (mirrors the create flow's
      // commit-on-submit): OBO writes the per-user token to the DB; legacy passthrough and the
      // client-forwarded modes (true_passthrough / oauth_delegate) keep it in sessionStorage and
      // never in the server row. M2M/static auth resolve server-side and need neither.
      if (oauthTokenResponse?.access_token) {
        const oauthMode = getMcpOAuthMode({
          auth_type: values.auth_type,
          oauth2_flow: isM2MFlow ? MCP_OAUTH2_FLOW_M2M : null,
          delegate_auth_to_upstream: Boolean(values.delegate_auth_to_upstream ?? mcpServer.delegate_auth_to_upstream),
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
          } else if (oauthMode === "passthrough" || isClientForwardedTokenMode(values.auth_type)) {
            const browserHeldToken = {
              access_token: oauthTokenResponse.access_token,
              expires_in: oauthTokenResponse.expires_in,
              token_type: oauthTokenResponse.token_type,
            };
            setToken(mcpServer.server_id, browserHeldToken, userID);
          }
        } catch (error: unknown) {
          const message = error instanceof Error ? error.message : "";
          toast.fromError("MCP Server updated, but failed to persist OAuth token" + (message ? `: ${message}` : ""));
          return;
        }
      }

      toast.success("MCP Server updated successfully");
      setAppMayNotMatchUpstream(false);
      onSuccess(updated);
    } catch (error: any) {
      toast.fromError("Failed to update MCP Server" + (error?.message ? `: ${error.message}` : ""));
    }
  };

  return (
    <Tabs defaultValue="server">
      <TabsList variant="line" className="grid h-auto w-full grid-cols-2 rounded-none border-b p-0">
        <TabsTrigger value="server" className="rounded-none py-2">
          Server Configuration
        </TabsTrigger>
        <TabsTrigger value="cost" className="rounded-none py-2">
          Cost Configuration
        </TabsTrigger>
      </TabsList>
      <div className="mt-6">
        <TabsContent value="server" keepMounted>
          <FormProvider {...form}>
            <MountedFormProvider value={{ control: form.control, registry }}>
              <form
                onSubmit={(event) => {
                  event.preventDefault();
                  void submitForm();
                }}
              >
                <MountedFormField
                  label="MCP Server Name"
                  name="server_name"
                  rules={{ validate: validatorRules({ validator: (_, value) => validateMCPServerName(value) }) }}
                >
                  {(control) => (
                    <Input
                      {...textControl(control)}
                      className="rounded-lg border-border focus:border-info focus:ring-ring"
                    />
                  )}
                </MountedFormField>
                <MountedFormField
                  label="Alias"
                  name="alias"
                  rules={{ validate: validatorRules({ validator: (_, value) => validateMCPServerName(value) }) }}
                >
                  {(control) => (
                    <Input
                      {...textControl(control)}
                      onChange={(event) => {
                        control.onChange(event);
                        setAliasManuallyEdited(true);
                      }}
                      className="rounded-lg border-border focus:border-info focus:ring-ring"
                    />
                  )}
                </MountedFormField>
                <MountedFormField label="Description" name="description">
                  {(control) => (
                    <Input
                      {...textControl(control)}
                      className="rounded-lg border-border focus:border-info focus:ring-ring"
                    />
                  )}
                </MountedFormField>
                <MCPLogoSelector value={logoUrl} onChange={setLogoUrl} />
                <MountedFormField
                  label="Transport Type"
                  name="transport"
                  required
                  rules={{ validate: { required: requiredRule("Transport Type is required") } }}
                >
                  {(control) => (
                    <Select
                      items={TRANSPORT_ITEMS}
                      value={(control.value as string | undefined) ?? null}
                      onValueChange={handleTransportSelected(control.onChange)}
                    >
                      <SelectTrigger {...selectTriggerControl(control)} className="w-full">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {TRANSPORT_ITEMS.map((item) => (
                          <SelectItem key={item.value} value={item.value}>
                            {item.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                </MountedFormField>

                {/* URL field - only for HTTP/SSE */}
                {isMCPTransport && (
                  <MountedFormField
                    label="MCP Server URL"
                    name="url"
                    required
                    rules={{
                      validate: {
                        required: requiredRule("Please enter a server URL"),
                        ...validatorRules({ validator: (_, value) => validateMCPServerUrl(value) }),
                      },
                    }}
                  >
                    {(control) => (
                      <Input
                        {...textControl(control)}
                        placeholder="https://your-mcp-server.com"
                        className="rounded-lg border-border focus:border-info focus:ring-ring"
                      />
                    )}
                  </MountedFormField>
                )}

                {/* OpenAPI Spec URL - only for OpenAPI transport */}
                {isOpenAPITransport && (
                  <MountedFormField
                    label={
                      <span className="text-sm font-medium text-foreground flex items-center">
                        OpenAPI Spec URL
                        <SimpleTooltip content="URL to an OpenAPI specification (JSON or YAML). MCP tools will be automatically generated from the API endpoints defined in the spec.">
                          <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                        </SimpleTooltip>
                      </span>
                    }
                    name="spec_path"
                    required
                    rules={{ validate: { required: requiredRule("Please enter an OpenAPI spec URL") } }}
                  >
                    {(control) => (
                      <Input
                        {...textControl(control)}
                        placeholder="https://petstore3.swagger.io/api/v3/openapi.json"
                        className="rounded-lg border-border focus:border-info focus:ring-ring"
                      />
                    )}
                  </MountedFormField>
                )}

                <MountedFormField
                  label={
                    <span className="text-sm font-medium text-foreground flex items-center">
                      Max Concurrent Requests (optional)
                      <SimpleTooltip content="Maximum number of tool calls LiteLLM will run against this server at the same time. Additional calls wait for a free slot. Leave blank for no limit.">
                        <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                      </SimpleTooltip>
                    </span>
                  }
                  name="max_concurrent_requests"
                >
                  {(control) => (
                    <Input
                      {...numberControl(control, 0)}
                      min={1}
                      step={1}
                      placeholder="e.g. 10"
                      className="w-full rounded-lg"
                    />
                  )}
                </MountedFormField>

                {/* Authentication - for HTTP, SSE, and OpenAPI */}
                {!isStdioTransport && (
                  <>
                    <MountedFormField
                      label="Authentication"
                      name="auth_type"
                      required
                      rules={{ validate: { required: requiredRule("Authentication is required") } }}
                    >
                      {(control) => (
                        <Select {...selectControl<string>(control)} items={AUTH_TYPE_ITEMS}>
                          <SelectTrigger {...selectTriggerControl(control)} className="w-full">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            {AUTH_TYPE_ITEMS.map((item) => (
                              <SelectItem key={item.value} value={item.value}>
                                {item.label}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      )}
                    </MountedFormField>
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
                  <div className="rounded-lg border border-border p-4 space-y-4">
                    <p className="text-sm text-muted-foreground">
                      Configure the stdio transport used to launch the MCP server process. You can either fill in the
                      fields below or paste a JSON configuration.
                    </p>

                    <MountedFormField
                      label="Command"
                      name="command"
                      required
                      rules={{ validate: { required: requiredRule("Please enter a command for stdio transport") } }}
                    >
                      {(control) => (
                        <Input
                          {...textControl(control)}
                          placeholder="e.g., npx"
                          className="rounded-lg border-border focus:border-info focus:ring-ring"
                        />
                      )}
                    </MountedFormField>

                    <MountedFormField label="Args" name="args">
                      {(control) => (
                        <MultiSelect
                          {...tagsControl(control)}
                          placeholder="Add args (press enter or comma)"
                          className="rounded-lg"
                        />
                      )}
                    </MountedFormField>

                    <MountedFormField
                      label="Environment (JSON object)"
                      name="env_json"
                      rules={{
                        validate: {
                          jsonObject: parsesAsJsonObject("Please enter valid JSON", "Env must be a JSON object"),
                        },
                      }}
                    >
                      {(control) => (
                        <Textarea
                          {...textControl(control)}
                          rows={6}
                          className="rounded-lg border-border focus:border-info focus:ring-ring font-mono text-sm"
                          placeholder={`{\n  \"KEY\": \"value\"\n}`}
                        />
                      )}
                    </MountedFormField>

                    {/* Optional JSON config (if provided, it overrides command/args/env on save) */}
                    <StdioConfiguration isVisible={true} required={false} />
                  </div>
                )}

                {!isStdioTransport && shouldShowAuthValueField && (
                  <MountedFormField
                    label={
                      <span className="text-sm font-medium text-foreground flex items-center">
                        Authentication Value
                        <SimpleTooltip content="Token, password, or header value to send with each request for the selected auth type.">
                          <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                        </SimpleTooltip>
                      </span>
                    }
                    name={["credentials", "auth_value"]}
                    rules={{ validate: { notWhitespace: notOnlyWhitespace("Authentication value cannot be empty") } }}
                  >
                    {(control) => (
                      <PasswordInput
                        {...textControl(control)}
                        placeholder="Enter token or secret (leave blank to keep existing)"
                        groupClassName="rounded-lg border-border focus:border-info focus:ring-ring"
                      />
                    )}
                  </MountedFormField>
                )}

                {!isStdioTransport && isOAuthAuthType && (
                  <>
                    {!oauthFlowTypeValue && !isDelegateAuth && (
                      <Alert variant="warning" className="mb-4 rounded-lg">
                        <TriangleAlert />
                        <AlertTitle>This server has no OAuth flow set</AlertTitle>
                        <AlertDescription>
                          Choose Machine-to-Machine (M2M) or Interactive (PKCE) so LiteLLM authenticates it the way you
                          intend, then save. Until it is set, LiteLLM falls back to interactive per-user auth and treats
                          a machine-to-machine credential shape conservatively.
                        </AlertDescription>
                      </Alert>
                    )}
                    <OAuthFormFields
                      isM2M={isM2MFlow}
                      isEditing
                      oauthFlow={{
                        startOAuthFlow,
                        status: oauthStatus,
                        error: oauthError,
                        tokenResponse: oauthTokenResponse,
                      }}
                    />
                  </>
                )}

                {!isStdioTransport && isTokenExchangeAuthType && <TokenExchangeFormFields isEditing />}

                {!isStdioTransport && isIdJagAuthType && <IdJagFormFields isEditing />}

                {!isStdioTransport && isAwsSigV4AuthType && (
                  <>
                    <p className="text-sm text-muted-foreground mb-2">
                      For MCP servers hosted on AWS Bedrock AgentCore.{" "}
                      <a
                        href="https://docs.litellm.ai/docs/mcp_aws_sigv4"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-info hover:text-info/80"
                      >
                        View docs &rarr;
                      </a>
                    </p>
                    <MountedFormField
                      label={
                        <span className="text-sm font-medium text-foreground flex items-center">
                          AWS Region
                          <SimpleTooltip content="AWS region for SigV4 signing (e.g., us-east-1)">
                            <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                          </SimpleTooltip>
                        </span>
                      }
                      name={["credentials", "aws_region_name"]}
                    >
                      {(control) => (
                        <Input
                          {...textControl(control)}
                          placeholder="us-east-1 (leave blank to keep existing)"
                          className="rounded-lg border-border focus:border-info focus:ring-ring"
                        />
                      )}
                    </MountedFormField>
                    <MountedFormField
                      label={
                        <span className="text-sm font-medium text-foreground flex items-center">
                          AWS Service Name
                          <SimpleTooltip content="AWS service name for SigV4 signing. Defaults to 'bedrock-agentcore'.">
                            <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                          </SimpleTooltip>
                        </span>
                      }
                      name={["credentials", "aws_service_name"]}
                    >
                      {(control) => (
                        <Input
                          {...textControl(control)}
                          placeholder="bedrock-agentcore (leave blank to keep existing)"
                          className="rounded-lg border-border focus:border-info focus:ring-ring"
                        />
                      )}
                    </MountedFormField>
                    <MountedFormField
                      label={
                        <span className="text-sm font-medium text-foreground flex items-center">
                          AWS Access Key ID
                          <SimpleTooltip content="Optional. If not provided, falls back to the boto3 credential chain (IAM role, env vars, etc.).">
                            <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                          </SimpleTooltip>
                        </span>
                      }
                      name={["credentials", "aws_access_key_id"]}
                    >
                      {(control) => (
                        <PasswordInput
                          {...textControl(control)}
                          placeholder="Leave blank to keep existing"
                          groupClassName="rounded-lg border-border focus:border-info focus:ring-ring"
                        />
                      )}
                    </MountedFormField>
                    <MountedFormField
                      label={
                        <span className="text-sm font-medium text-foreground flex items-center">
                          AWS Secret Access Key
                          <SimpleTooltip content="Optional. Required if AWS Access Key ID is provided.">
                            <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                          </SimpleTooltip>
                        </span>
                      }
                      name={["credentials", "aws_secret_access_key"]}
                    >
                      {(control) => (
                        <PasswordInput
                          {...textControl(control)}
                          placeholder="Leave blank to keep existing"
                          groupClassName="rounded-lg border-border focus:border-info focus:ring-ring"
                        />
                      )}
                    </MountedFormField>
                    <MountedFormField
                      label={
                        <span className="text-sm font-medium text-foreground flex items-center">
                          AWS Session Token
                          <SimpleTooltip content="Optional. Only needed for temporary STS credentials.">
                            <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                          </SimpleTooltip>
                        </span>
                      }
                      name={["credentials", "aws_session_token"]}
                    >
                      {(control) => (
                        <PasswordInput
                          {...textControl(control)}
                          placeholder="Leave blank to keep existing"
                          groupClassName="rounded-lg border-border focus:border-info focus:ring-ring"
                        />
                      )}
                    </MountedFormField>
                    <MountedFormField
                      label={
                        <span className="text-sm font-medium text-foreground flex items-center">
                          AWS Role ARN
                          <SimpleTooltip content="Optional. IAM role ARN to assume via STS before signing. If set, LiteLLM calls sts:AssumeRole to get temporary credentials.">
                            <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                          </SimpleTooltip>
                        </span>
                      }
                      name={["credentials", "aws_role_name"]}
                    >
                      {(control) => (
                        <Input
                          {...textControl(control)}
                          placeholder="Leave blank to keep existing"
                          className="rounded-lg border-border focus:border-info focus:ring-ring"
                        />
                      )}
                    </MountedFormField>
                    <MountedFormField
                      label={
                        <span className="text-sm font-medium text-foreground flex items-center">
                          AWS Session Name
                          <SimpleTooltip content="Optional. Session name for the AssumeRole call — appears in CloudTrail logs. Auto-generated if omitted.">
                            <Info className="ml-2 size-4 text-info hover:text-info/80 cursor-help" />
                          </SimpleTooltip>
                        </span>
                      }
                      name={["credentials", "aws_session_name"]}
                    >
                      {(control) => (
                        <Input
                          {...textControl(control)}
                          placeholder="Leave blank to keep existing"
                          className="rounded-lg border-border focus:border-info focus:ring-ring"
                        />
                      )}
                    </MountedFormField>
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
                    mountedAuthType={authType}
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
                  <Button variant="outline" onClick={onCancel}>
                    Cancel
                  </Button>
                  <Button type="submit">Save Changes</Button>
                </div>
              </form>
            </MountedFormProvider>
          </FormProvider>
        </TabsContent>

        <TabsContent value="cost" keepMounted>
          <div className="space-y-6">
            <MCPServerCostConfig value={costConfig} onChange={setCostConfig} tools={tools} disabled={isLoadingTools} />

            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={onCancel}>
                Cancel
              </Button>
              <Button onClick={() => void submitForm()}>Save Changes</Button>
            </div>
          </div>
        </TabsContent>
      </div>
    </Tabs>
  );
};

export default MCPServerEdit;
