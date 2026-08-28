import {
  getGuardrailInfo,
  getGuardrailProviderSpecificParams,
  getGuardrailUISettings,
  updateGuardrailCall,
} from "@/components/networking";
import { copyToClipboard as utilCopyToClipboard } from "@/utils/dataUtils";

import { ArrowLeft, Ban, CheckIcon, Code, CopyIcon, EyeOff, Info } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import React, { useCallback, useEffect, useLayoutEffect, useState } from "react";
import { useForm } from "react-hook-form";
import { toast } from "@/lib/toast";
import { Logo } from "@/components/molecules/logo/Logo";
import { FieldGroup } from "@/components/ui/field";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { SimpleTooltip, TooltipProvider } from "@/components/ui/tooltip";
import {
  asText,
  GuardrailField,
  labelWithHint,
  readRecord,
  requiredRule,
  type GuardrailFormValues,
  SkipMessageSelect,
} from "./GuardrailFormField";
import ContentFilterManager, { formatContentFilterDataForAPI } from "./content_filter/ContentFilterManager";
import CustomCodeModal, { EditGuardrailData } from "./custom_code/CustomCodeModal";
import {
  formatGuardrailMode,
  getGuardrailLogoAndName,
  guardrail_provider_map,
  skipSystemMessageToChoice,
  skipToolMessageToChoice,
  type SkipSystemMessageChoice,
  type SkipToolMessageChoice,
} from "./guardrail_info_helpers";
import GuardrailOptionalParams from "./guardrail_optional_params";
import GuardrailProviderFields from "./guardrail_provider_fields";
import PiiConfiguration from "./pii_configuration";
import ToolPermissionRulesEditor, { ToolPermissionConfig } from "./tool_permission/ToolPermissionRulesEditor";

const DEFAULT_ON_ITEMS = [
  { label: "Yes", value: true },
  { label: "No", value: false },
];

const SectionHeading: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <div className="my-6 flex items-center gap-3">
    <span className="shrink-0 text-sm font-medium text-foreground">{children}</span>
    <Separator className="flex-1" />
  </div>
);

export interface GuardrailInfoProps {
  guardrailId: string;
  onClose: () => void;
  accessToken: string | null;
  isAdmin: boolean;
}

const GuardrailInfoView: React.FC<GuardrailInfoProps> = ({ guardrailId, onClose, accessToken, isAdmin }) => {
  const [guardrailData, setGuardrailData] = useState<any>(null);
  const [guardrailProviderSpecificParams, setGuardrailProviderSpecificParams] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [isEditing, setIsEditing] = useState(false);
  const form = useForm<GuardrailFormValues>({ defaultValues: {} });
  const [selectedPiiEntities, setSelectedPiiEntities] = useState<string[]>([]);
  const [selectedPiiActions, setSelectedPiiActions] = useState<{ [key: string]: string }>({});
  const [guardrailSettings, setGuardrailSettings] = useState<{
    supported_entities: string[];
    supported_actions: string[];
    pii_entity_categories: Array<{
      category: string;
      entities: string[];
    }>;
    supported_modes: string[];
    content_filter_settings?: {
      prebuilt_patterns: Array<{
        name: string;
        display_name: string;
        category: string;
        description: string;
      }>;
      pattern_categories: string[];
      supported_actions: string[];
      content_categories?: Array<{
        name: string;
        display_name: string;
        description: string;
        default_action: string;
      }>;
    };
  } | null>(null);
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});
  const [hasUnsavedContentFilterChanges, setHasUnsavedContentFilterChanges] = useState(false);
  const emptyToolPermissionConfig: ToolPermissionConfig = {
    rules: [],
    default_action: "deny",
    on_disallowed_action: "block",
    violation_message_template: "",
  };
  const [toolPermissionConfig, setToolPermissionConfig] = useState<ToolPermissionConfig>(emptyToolPermissionConfig);
  const [toolPermissionDirty, setToolPermissionDirty] = useState(false);
  const [customCodeModalVisible, setCustomCodeModalVisible] = useState(false);

  // Content Filter data ref (managed by ContentFilterManager)
  const contentFilterDataRef = React.useRef<{
    patterns: any[];
    blockedWords: any[];
    categories: any[];
    competitorIntentEnabled?: boolean;
    competitorIntentConfig?: any;
  }>({
    patterns: [],
    blockedWords: [],
    categories: [],
  });

  // Memoize onDataChange callback to prevent unnecessary re-renders
  const handleContentFilterDataChange = useCallback(
    (
      patterns: any[],
      blockedWords: any[],
      categories: any[],
      competitorIntentEnabled?: boolean,
      competitorIntentConfig?: any,
    ) => {
      contentFilterDataRef.current = {
        patterns,
        blockedWords,
        categories: categories || [],
        competitorIntentEnabled,
        competitorIntentConfig,
      };
    },
    [],
  );

  const fetchGuardrailInfo = async () => {
    try {
      setLoading(true);
      if (!accessToken) return;
      const response = await getGuardrailInfo(accessToken, guardrailId);
      setGuardrailData(response);

      // Initialize PII configuration from guardrail data
      if (response.litellm_params?.pii_entities_config) {
        const piiConfig = response.litellm_params.pii_entities_config;

        // Clear previous selections
        setSelectedPiiEntities([]);
        setSelectedPiiActions({});

        // Only if there are entities configured
        if (Object.keys(piiConfig).length > 0) {
          const entities: string[] = [];
          const actions: { [key: string]: string } = {};

          Object.entries(piiConfig).forEach(([entity, action]: [string, any]) => {
            entities.push(entity);
            actions[entity] = typeof action === "string" ? action : "MASK";
          });

          setSelectedPiiEntities(entities);
          setSelectedPiiActions(actions);
        }
      } else {
        // Clear selections if no PII config exists
        setSelectedPiiEntities([]);
        setSelectedPiiActions({});
      }
    } catch (error) {
      toast.fromError("Failed to load guardrail information");
      console.error("Error fetching guardrail info:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchGuardrailProviderSpecificParams = async () => {
    try {
      if (!accessToken) return;
      const response = await getGuardrailProviderSpecificParams(accessToken);
      setGuardrailProviderSpecificParams(response);
    } catch (error) {
      console.error("Error fetching guardrail provider specific params:", error);
    }
  };

  const fetchGuardrailUISettings = async () => {
    try {
      if (!accessToken) return;
      const uiSettings = await getGuardrailUISettings(accessToken);
      setGuardrailSettings(uiSettings);
    } catch (error) {
      console.error("Error fetching guardrail UI settings:", error);
    }
  };

  useEffect(() => {
    fetchGuardrailProviderSpecificParams();
  }, [accessToken]);

  useEffect(() => {
    fetchGuardrailInfo();
    fetchGuardrailUISettings();
  }, [guardrailId, accessToken]);

  // Reset form when guardrail data or provider params change. Only the names this form actually
  // binds are seeded: an unbound key would otherwise be submitted as if the user had set it.
  useEffect(() => {
    if (!guardrailData) return;
    form.setValue("guardrail_name", guardrailData.guardrail_name);
    form.setValue("default_on", guardrailData.litellm_params?.default_on);
    form.setValue(
      "skip_system_message_choice",
      skipSystemMessageToChoice(guardrailData.litellm_params?.skip_system_message_in_guardrail),
    );
    form.setValue(
      "skip_tool_message_choice",
      skipToolMessageToChoice(guardrailData.litellm_params?.skip_tool_message_in_guardrail),
    );
    form.setValue(
      "guardrail_info",
      guardrailData.guardrail_info ? JSON.stringify(guardrailData.guardrail_info, null, 2) : "",
    );
    if (guardrailData.litellm_params?.optional_params) {
      form.setValue("optional_params", guardrailData.litellm_params.optional_params);
    }
  }, [guardrailData, guardrailProviderSpecificParams, form]);

  const resetToolPermissionEditor = useCallback(() => {
    if (guardrailData?.litellm_params?.guardrail === "tool_permission") {
      setToolPermissionConfig({
        rules: (guardrailData.litellm_params?.rules as ToolPermissionConfig["rules"]) || [],
        default_action: (
          (guardrailData.litellm_params?.default_action || "deny") as ToolPermissionConfig["default_action"]
        ).toLowerCase() as ToolPermissionConfig["default_action"],
        on_disallowed_action: (
          (guardrailData.litellm_params?.on_disallowed_action ||
            "block") as ToolPermissionConfig["on_disallowed_action"]
        ).toLowerCase() as ToolPermissionConfig["on_disallowed_action"],
        violation_message_template: guardrailData.litellm_params?.violation_message_template || "",
      });
    } else {
      setToolPermissionConfig(emptyToolPermissionConfig);
    }
    setToolPermissionDirty(false);
  }, [guardrailData]);

  useEffect(() => {
    resetToolPermissionEditor();
  }, [resetToolPermissionEditor]);

  const handlePiiEntitySelect = (entity: string) => {
    setSelectedPiiEntities((prev) => {
      if (prev.includes(entity)) {
        return prev.filter((e) => e !== entity);
      } else {
        return [...prev, entity];
      }
    });
  };

  const handlePiiActionSelect = (entity: string, action: string) => {
    setSelectedPiiActions((prev) => ({
      ...prev,
      [entity]: action,
    }));
  };

  const handleGuardrailUpdate = async (values: GuardrailFormValues) => {
    try {
      if (!accessToken) return;

      // Prepare update data object - only include changed fields
      const updateData: any = {
        litellm_params: {},
      };

      // Only include guardrail_name if it has changed
      if (values.guardrail_name !== guardrailData.guardrail_name) {
        updateData.guardrail_name = values.guardrail_name;
      }

      // Only include default_on if it has changed
      if (values.default_on !== guardrailData.litellm_params?.default_on) {
        updateData.litellm_params.default_on = values.default_on;
      }

      const prevSkipChoice = skipSystemMessageToChoice(guardrailData.litellm_params?.skip_system_message_in_guardrail);
      const nextSkipChoice = values.skip_system_message_choice as SkipSystemMessageChoice | undefined;
      if (nextSkipChoice !== undefined && nextSkipChoice !== prevSkipChoice) {
        if (nextSkipChoice === "inherit") {
          updateData.litellm_params.skip_system_message_in_guardrail = null;
        } else if (nextSkipChoice === "yes") {
          updateData.litellm_params.skip_system_message_in_guardrail = true;
        } else {
          updateData.litellm_params.skip_system_message_in_guardrail = false;
        }
      }

      const prevSkipToolChoice = skipToolMessageToChoice(guardrailData.litellm_params?.skip_tool_message_in_guardrail);
      const nextSkipToolChoice = values.skip_tool_message_choice as SkipToolMessageChoice | undefined;
      if (nextSkipToolChoice !== undefined && nextSkipToolChoice !== prevSkipToolChoice) {
        if (nextSkipToolChoice === "inherit") {
          updateData.litellm_params.skip_tool_message_in_guardrail = null;
        } else if (nextSkipToolChoice === "yes") {
          updateData.litellm_params.skip_tool_message_in_guardrail = true;
        } else {
          updateData.litellm_params.skip_tool_message_in_guardrail = false;
        }
      }

      // Only include guardrail_info if it has changed
      const originalGuardrailInfo = guardrailData.guardrail_info;
      const newGuardrailInfo = values.guardrail_info ? JSON.parse(asText(values.guardrail_info)) : undefined;
      if (JSON.stringify(originalGuardrailInfo) !== JSON.stringify(newGuardrailInfo)) {
        updateData.guardrail_info = newGuardrailInfo;
      }

      // Only add PII entities config if there are changes
      const originalPiiConfig = guardrailData.litellm_params?.pii_entities_config || {};
      const newPiiEntitiesConfig: { [key: string]: string } = {};

      selectedPiiEntities.forEach((entity) => {
        newPiiEntitiesConfig[entity] = selectedPiiActions[entity] || "MASK";
      });

      // Only update if PII config has changed
      if (JSON.stringify(originalPiiConfig) !== JSON.stringify(newPiiEntitiesConfig)) {
        updateData.litellm_params.pii_entities_config = newPiiEntitiesConfig;
      }

      // Only add Content Filter patterns if there are changes
      if (guardrailData.litellm_params?.guardrail === "litellm_content_filter" && hasUnsavedContentFilterChanges) {
        const formattedData = formatContentFilterDataForAPI(
          contentFilterDataRef.current.patterns || [],
          contentFilterDataRef.current.blockedWords || [],
          contentFilterDataRef.current.categories || [],
          contentFilterDataRef.current.competitorIntentEnabled,
          contentFilterDataRef.current.competitorIntentConfig,
        );

        updateData.litellm_params.patterns = formattedData.patterns;
        updateData.litellm_params.blocked_words = formattedData.blocked_words;
        updateData.litellm_params.categories = formattedData.categories;
        updateData.litellm_params.competitor_intent_config = formattedData.competitor_intent_config ?? null;
      }

      if (guardrailData.litellm_params?.guardrail === "tool_permission") {
        const originalRules = guardrailData.litellm_params?.rules || [];
        const currentRules = toolPermissionConfig.rules || [];
        const rulesChanged = JSON.stringify(originalRules) !== JSON.stringify(currentRules);

        const originalDefault = (guardrailData.litellm_params?.default_action || "deny").toLowerCase();
        const currentDefault = (toolPermissionConfig.default_action || "deny").toLowerCase();
        const defaultChanged = originalDefault !== currentDefault;

        const originalOnDisallowed = (guardrailData.litellm_params?.on_disallowed_action || "block").toLowerCase();
        const currentOnDisallowed = (toolPermissionConfig.on_disallowed_action || "block").toLowerCase();
        const onDisallowedChanged = originalOnDisallowed !== currentOnDisallowed;

        const originalMessage = guardrailData.litellm_params?.violation_message_template || "";
        const currentMessage = toolPermissionConfig.violation_message_template || "";
        const messageChanged = originalMessage !== currentMessage;

        if (toolPermissionDirty || rulesChanged || defaultChanged || onDisallowedChanged || messageChanged) {
          updateData.litellm_params.rules = currentRules;
          updateData.litellm_params.default_action = currentDefault;
          updateData.litellm_params.on_disallowed_action = currentOnDisallowed;
          updateData.litellm_params.violation_message_template = currentMessage || null;
        }
      }

      /******************************
       * Add provider-specific params (reusing logic from add_guardrail_form.tsx)
       * ----------------------------------
       * The backend exposes exactly which extra parameters a provider
       * accepts via `/guardrails/ui/provider_specific_params`.
       * Instead of copying every unknown form field, we fetch the list for
       * the selected provider and ONLY pass those recognised params.
       ******************************/

      // Get the current provider from the guardrail data
      const currentProvider = Object.keys(guardrail_provider_map).find(
        (key) => guardrail_provider_map[key] === guardrailData.litellm_params?.guardrail,
      );

      // Use pre-fetched provider params to copy recognised params
      const isToolPermissionGuardrail = guardrailData.litellm_params?.guardrail === "tool_permission";
      if (guardrailProviderSpecificParams && currentProvider && !isToolPermissionGuardrail) {
        const providerKey = guardrail_provider_map[currentProvider]?.toLowerCase();
        const providerSpecificParams = guardrailProviderSpecificParams[providerKey] || {};

        const allowedParams = new Set<string>();

        // Add root-level parameters (like api_key, api_base, api_version)
        Object.keys(providerSpecificParams).forEach((paramName) => {
          if (paramName !== "optional_params") {
            allowedParams.add(paramName);
          }
        });

        // Add nested parameters from optional_params.fields
        if (providerSpecificParams.optional_params && providerSpecificParams.optional_params.fields) {
          Object.keys(providerSpecificParams.optional_params.fields).forEach((paramName) => {
            allowedParams.add(paramName);
          });
        }

        allowedParams.forEach((paramName) => {
          if (paramName === "patterns" || paramName === "blocked_words" || paramName === "categories") {
            return;
          }
          // Check for both direct parameter name and nested optional_params object
          const directValue = values[paramName];
          const paramValue =
            directValue === undefined || directValue === null || directValue === ""
              ? readRecord(values.optional_params, paramName)
              : directValue;

          // Get the original value for comparison
          const originalValue = guardrailData.litellm_params?.[paramName];

          // Check if the value has changed from the original
          const hasChanged = JSON.stringify(paramValue) !== JSON.stringify(originalValue);

          // Include if value has changed and has a meaningful value, OR if user explicitly cleared a value
          if (hasChanged) {
            if (paramValue !== undefined && paramValue !== null && paramValue !== "") {
              // User set a new value
              updateData.litellm_params[paramName] = paramValue;
            } else if (originalValue !== undefined && originalValue !== null && originalValue !== "") {
              // User cleared an existing value - set to null to indicate removal
              updateData.litellm_params[paramName] = null;
            }
          }
        });
      }

      // Remove empty litellm_params object if no parameters were changed
      if (Object.keys(updateData.litellm_params).length === 0) {
        delete updateData.litellm_params;
      }

      // Only proceed with update if there are actual changes
      if (Object.keys(updateData).length === 0) {
        toast.info("No changes detected");
        setIsEditing(false);
        return;
      }

      await updateGuardrailCall(accessToken, guardrailId, updateData);
      toast.success("Guardrail updated successfully");
      setHasUnsavedContentFilterChanges(false);
      fetchGuardrailInfo();
      setIsEditing(false);
    } catch (error) {
      console.error("Error updating guardrail:", error);
      toast.fromError("Failed to update guardrail");
    }
  };

  // antd re-read onFinish at validation-resolution time, so a submit fired by the same click that
  // updated state saw that state; a captured handler would not.
  const submitRef = React.useRef(handleGuardrailUpdate);
  useLayoutEffect(() => {
    submitRef.current = handleGuardrailUpdate;
  });
  const submitLatest = useCallback((values: GuardrailFormValues) => submitRef.current(values), []);

  if (loading) {
    return <div className="p-4">Loading...</div>;
  }

  if (!guardrailData) {
    return <div className="p-4">Guardrail not found</div>;
  }

  // Format date helper function
  const formatDate = (dateString?: string) => {
    if (!dateString) return "-";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  // Format the provider display name and logo
  const { logo, displayName } = getGuardrailLogoAndName(guardrailData.litellm_params?.guardrail || "");

  const copyToClipboard = async (text: string | null | undefined, key: string) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
    }
  };

  const isConfigGuardrail = guardrailData.guardrail_definition_location === "config";

  return (
    <div className="p-4">
      <div>
        <Button variant="ghost" onClick={onClose} className="mb-4">
          <ArrowLeft className="w-4 h-4" />
          Back to Guardrails
        </Button>
        <h1 className="text-2xl font-semibold">{guardrailData.guardrail_name || "Unnamed Guardrail"}</h1>
        <div className="flex items-center cursor-pointer">
          <p className="text-muted-foreground font-mono">{guardrailData.guardrail_id}</p>

          <Button
            variant="ghost"
            size="icon-xs"
            onClick={() => copyToClipboard(guardrailData.guardrail_id, "guardrail-id")}
            className={`left-2 z-raised transition-all duration-200 ${
              copiedStates["guardrail-id"]
                ? "text-success bg-success/10 border-success/20"
                : "text-muted-foreground hover:text-foreground hover:bg-muted"
            }`}
          >
            {copiedStates["guardrail-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
          </Button>
        </div>
      </div>

      <Tabs defaultValue="overview">
        <TabsList variant="line" className="mb-4 h-auto w-full justify-start rounded-none border-b p-0">
          <TabsTrigger value="overview" className="flex-none rounded-none px-4 py-2">
            Overview
          </TabsTrigger>
          {isAdmin && (
            <TabsTrigger value="settings" className="flex-none rounded-none px-4 py-2">
              Settings
            </TabsTrigger>
          )}
        </TabsList>

        <div>
          {/* Overview Panel */}
          <TabsContent value="overview" keepMounted>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              <Card className="block p-6">
                <p>Provider</p>
                <div className="mt-2 flex items-center space-x-2">
                  <Logo src={logo} label={displayName} className="w-6 h-6" />
                  <h3 className="text-lg font-medium">{displayName}</h3>
                </div>
              </Card>

              <Card className="block p-6">
                <p>Mode</p>
                <div className="mt-2">
                  <h3 className="text-lg font-medium">
                    {formatGuardrailMode(guardrailData.litellm_params?.mode) || "-"}
                  </h3>
                  <Badge variant={guardrailData.litellm_params?.default_on ? "secondary" : "outline"}>
                    {guardrailData.litellm_params?.default_on ? "Default On" : "Default Off"}
                  </Badge>
                </div>
              </Card>

              <Card className="block p-6">
                <p>Created At</p>
                <div className="mt-2">
                  <h3 className="text-lg font-medium">{formatDate(guardrailData.created_at)}</h3>
                  <p>Last Updated: {formatDate(guardrailData.updated_at)}</p>
                </div>
              </Card>
            </div>

            {guardrailData.litellm_params?.pii_entities_config &&
              Object.keys(guardrailData.litellm_params.pii_entities_config).length > 0 && (
                <Card className="block mt-6 p-6">
                  <div className="flex justify-between items-center">
                    <p className="font-medium">PII Protection</p>
                    <Badge variant="secondary">
                      {Object.keys(guardrailData.litellm_params.pii_entities_config).length} PII entities configured
                    </Badge>
                  </div>
                </Card>
              )}

            {guardrailData.litellm_params?.pii_entities_config &&
              Object.keys(guardrailData.litellm_params.pii_entities_config).length > 0 && (
                <Card className="block mt-6 p-6">
                  <p className="mb-4 text-lg font-semibold">PII Entity Configuration</p>
                  <div className="border rounded-lg overflow-hidden shadow-xs">
                    <div className="bg-muted px-5 py-3 border-b flex">
                      <p className="flex-1 font-semibold text-foreground">Entity Type</p>
                      <p className="flex-1 font-semibold text-foreground">Configuration</p>
                    </div>
                    <div className="max-h-[400px] overflow-y-auto">
                      {Object.entries(guardrailData.litellm_params?.pii_entities_config).map(([key, value]) => (
                        <div key={key} className="px-5 py-3 flex border-b hover:bg-muted/50 transition-colors">
                          <p className="flex-1 font-medium text-foreground">{key}</p>
                          <p className="flex-1">
                            <span
                              className={`inline-flex items-center gap-1.5 ${
                                value === "MASK" ? "text-info" : "text-destructive"
                              }`}
                            >
                              {value === "MASK" ? <EyeOff className="size-3.5" /> : <Ban className="size-3.5" />}
                              {String(value)}
                            </span>
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </Card>
              )}

            {guardrailData.litellm_params?.guardrail === "tool_permission" && (
              <Card className="block mt-6 p-6">
                <ToolPermissionRulesEditor value={toolPermissionConfig} disabled />
              </Card>
            )}

            {/* Custom Code Display */}
            {guardrailData.litellm_params?.guardrail === "custom_code" && guardrailData.litellm_params?.custom_code && (
              <Card className="block mt-6 p-6">
                <div className="flex justify-between items-center mb-4">
                  <div className="flex items-center gap-2">
                    <Code className="text-info" />
                    <p className="font-medium text-lg">Custom Code</p>
                  </div>
                  {isAdmin && !isConfigGuardrail && (
                    <Button variant="outline" size="sm" onClick={() => setCustomCodeModalVisible(true)}>
                      <Code />
                      Edit Code
                    </Button>
                  )}
                </div>
                <div className="relative rounded-lg overflow-hidden border border-gray-700 bg-[#1e1e1e]">
                  <pre
                    className="p-4 text-sm text-gray-200 overflow-x-auto"
                    style={{ fontFamily: "'Fira Code', 'Monaco', 'Consolas', monospace" }}
                  >
                    <code>{guardrailData.litellm_params.custom_code}</code>
                  </pre>
                </div>
              </Card>
            )}

            {/* Content Filter Configuration Display */}
            <ContentFilterManager
              guardrailData={guardrailData}
              guardrailSettings={guardrailSettings}
              isEditing={false}
              accessToken={accessToken}
            />
          </TabsContent>

          {/* Settings Panel (only for admins) */}
          {isAdmin && (
            <TabsContent value="settings" keepMounted>
              <Card className="block p-6">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-medium">Guardrail Settings</h3>
                  {isConfigGuardrail && (
                    <SimpleTooltip content="Guardrail is defined in the config file and cannot be edited.">
                      <Info role="img" aria-label="Config guardrail details" className="size-4 text-muted-foreground" />
                    </SimpleTooltip>
                  )}
                  {!isEditing &&
                    !isConfigGuardrail &&
                    (guardrailData.litellm_params?.guardrail === "custom_code" ? (
                      <Button variant="outline" onClick={() => setCustomCodeModalVisible(true)}>
                        <Code />
                        Edit Code
                      </Button>
                    ) : (
                      <Button variant="outline" onClick={() => setIsEditing(true)}>
                        Edit Settings
                      </Button>
                    ))}
                </div>

                {isEditing ? (
                  <TooltipProvider>
                    {/* eslint-disable-next-line react-hooks/refs -- latest-handler ref, read only after validation resolves */}
                    <form onSubmit={form.handleSubmit(submitLatest)}>
                      <FieldGroup>
                        <GuardrailField
                          control={form.control}
                          name="guardrail_name"
                          label="Guardrail Name"
                          rules={requiredRule("Please input a guardrail name")}
                        >
                          {({ ref, value, ...field }) => (
                            <Input {...field} ref={ref} value={asText(value)} placeholder="Enter guardrail name" />
                          )}
                        </GuardrailField>

                        <GuardrailField control={form.control} name="default_on" label="Default On">
                          {({ id, value, onChange, "aria-invalid": ariaInvalid, "aria-describedby": describedBy }) => (
                            <Select
                              items={DEFAULT_ON_ITEMS}
                              value={typeof value === "boolean" ? value : null}
                              onValueChange={(next: boolean | null) => onChange(next)}
                            >
                              <SelectTrigger
                                id={id}
                                aria-invalid={ariaInvalid}
                                aria-describedby={describedBy}
                                className="w-full"
                              >
                                <SelectValue placeholder="Select an option" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value={true}>Yes</SelectItem>
                                <SelectItem value={false}>No</SelectItem>
                              </SelectContent>
                            </Select>
                          )}
                        </GuardrailField>

                        <GuardrailField
                          control={form.control}
                          name="skip_system_message_choice"
                          label={labelWithHint(
                            "Skip system messages in guardrail",
                            "Unified guardrails: omit role: system from guardrail input (LLM still gets full messages). Use global default follows litellm_settings.skip_system_message_in_guardrail.",
                          )}
                        >
                          {(fieldControl) => <SkipMessageSelect control={fieldControl} />}
                        </GuardrailField>

                        <GuardrailField
                          control={form.control}
                          name="skip_tool_message_choice"
                          label={labelWithHint(
                            "Skip tool messages in guardrail",
                            "Unified guardrails: omit role: tool from guardrail input (LLM still gets full messages). Use global default follows litellm_settings.skip_tool_message_in_guardrail.",
                          )}
                        >
                          {(fieldControl) => <SkipMessageSelect control={fieldControl} />}
                        </GuardrailField>
                        {guardrailData.litellm_params?.guardrail === "presidio" && (
                          <>
                            <SectionHeading>PII Protection</SectionHeading>
                            <div className="mb-6">
                              {guardrailSettings && (
                                <PiiConfiguration
                                  entities={guardrailSettings.supported_entities}
                                  actions={guardrailSettings.supported_actions}
                                  selectedEntities={selectedPiiEntities}
                                  selectedActions={selectedPiiActions}
                                  onEntitySelect={handlePiiEntitySelect}
                                  onActionSelect={handlePiiActionSelect}
                                  entityCategories={guardrailSettings.pii_entity_categories}
                                />
                              )}
                            </div>
                          </>
                        )}

                        <ContentFilterManager
                          guardrailData={guardrailData}
                          guardrailSettings={guardrailSettings}
                          isEditing={true}
                          accessToken={accessToken}
                          onDataChange={handleContentFilterDataChange}
                          onUnsavedChanges={setHasUnsavedContentFilterChanges}
                        />

                        {(guardrailData.litellm_params?.guardrail === "tool_permission" ||
                          guardrailProviderSpecificParams) && <SectionHeading>Provider Settings</SectionHeading>}

                        {guardrailData.litellm_params?.guardrail === "tool_permission" ? (
                          <ToolPermissionRulesEditor value={toolPermissionConfig} onChange={setToolPermissionConfig} />
                        ) : (
                          <>
                            {/* Provider-specific fields */}
                            <GuardrailProviderFields
                              selectedProvider={
                                Object.keys(guardrail_provider_map).find(
                                  (key) => guardrail_provider_map[key] === guardrailData.litellm_params?.guardrail,
                                ) || null
                              }
                              control={form.control}
                              accessToken={accessToken}
                              providerParams={guardrailProviderSpecificParams}
                              value={guardrailData.litellm_params}
                            />

                            {/* Optional parameters */}
                            {guardrailProviderSpecificParams &&
                              (() => {
                                const currentProvider = Object.keys(guardrail_provider_map).find(
                                  (key) => guardrail_provider_map[key] === guardrailData.litellm_params?.guardrail,
                                );
                                if (!currentProvider) return null;

                                const providerKey = guardrail_provider_map[currentProvider]?.toLowerCase();
                                const providerFields = guardrailProviderSpecificParams[providerKey];

                                if (!providerFields || !providerFields.optional_params) return null;

                                return (
                                  <GuardrailOptionalParams
                                    optionalParams={providerFields.optional_params}
                                    parentFieldKey="optional_params"
                                    control={form.control}
                                    values={guardrailData.litellm_params}
                                  />
                                );
                              })()}
                          </>
                        )}

                        <SectionHeading>Advanced Settings</SectionHeading>
                        <GuardrailField control={form.control} name="guardrail_info" label="Guardrail Information">
                          {({ ref, value, ...field }) => (
                            <Textarea {...field} ref={ref} value={asText(value)} rows={5} />
                          )}
                        </GuardrailField>

                        <div className="mt-6 flex justify-end gap-2">
                          <Button
                            type="button"
                            variant="outline"
                            onClick={() => {
                              setIsEditing(false);
                              setHasUnsavedContentFilterChanges(false);
                              resetToolPermissionEditor();
                            }}
                          >
                            Cancel
                          </Button>
                          <Button type="submit">Save Changes</Button>
                        </div>
                      </FieldGroup>
                    </form>
                  </TooltipProvider>
                ) : (
                  <div className="space-y-4">
                    <div>
                      <p className="font-medium">Guardrail ID</p>
                      <div className="font-mono">{guardrailData.guardrail_id}</div>
                    </div>
                    <div>
                      <p className="font-medium">Guardrail Name</p>
                      <div>{guardrailData.guardrail_name || "Unnamed Guardrail"}</div>
                    </div>
                    <div>
                      <p className="font-medium">Provider</p>
                      <div>{displayName}</div>
                    </div>
                    <div>
                      <p className="font-medium">Mode</p>
                      <div>{formatGuardrailMode(guardrailData.litellm_params?.mode) || "-"}</div>
                    </div>
                    <div>
                      <p className="font-medium">Default On</p>
                      <Badge variant={guardrailData.litellm_params?.default_on ? "secondary" : "outline"}>
                        {guardrailData.litellm_params?.default_on ? "Yes" : "No"}
                      </Badge>
                    </div>

                    {guardrailData.litellm_params?.pii_entities_config &&
                      Object.keys(guardrailData.litellm_params.pii_entities_config).length > 0 && (
                        <div>
                          <p className="font-medium">PII Protection</p>
                          <div className="mt-2">
                            <Badge variant="secondary">
                              {Object.keys(guardrailData.litellm_params.pii_entities_config).length} PII entities
                              configured
                            </Badge>
                          </div>
                        </div>
                      )}

                    <div>
                      <p className="font-medium">Created At</p>
                      <div>{formatDate(guardrailData.created_at)}</div>
                    </div>
                    <div>
                      <p className="font-medium">Last Updated</p>
                      <div>{formatDate(guardrailData.updated_at)}</div>
                    </div>

                    {guardrailData.litellm_params?.guardrail === "tool_permission" && (
                      <ToolPermissionRulesEditor value={toolPermissionConfig} disabled />
                    )}
                  </div>
                )}
              </Card>
            </TabsContent>
          )}
        </div>
      </Tabs>

      {/* Custom Code Editor Modal */}
      <CustomCodeModal
        visible={customCodeModalVisible}
        onClose={() => setCustomCodeModalVisible(false)}
        onSuccess={() => {
          setCustomCodeModalVisible(false);
          fetchGuardrailInfo();
        }}
        accessToken={accessToken}
        editData={
          guardrailData
            ? ({
                guardrail_id: guardrailData.guardrail_id,
                guardrail_name: guardrailData.guardrail_name,
                litellm_params: guardrailData.litellm_params,
              } as EditGuardrailData)
            : null
        }
      />
    </div>
  );
};

export default GuardrailInfoView;
