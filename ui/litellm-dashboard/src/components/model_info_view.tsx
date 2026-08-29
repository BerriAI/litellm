import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { useModelHub, useModelsInfo } from "@/app/(dashboard)/hooks/models/useModels";
import { useQueryClient } from "@tanstack/react-query";
import { transformModelData } from "@/app/(dashboard)/models-and-endpoints/utils/modelDataTransformer";
import { KeyIcon, RefreshIcon, TrashIcon } from "@heroicons/react/outline";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { applyPtuModelInfo } from "../utils/ptuModelInfo";
import { usePtuCostAttributionEnabled } from "@/app/(dashboard)/hooks/uiSettings/usePtuCostAttributionEnabled";
import { ArrowLeft, CheckIcon, CopyIcon, Info } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { copyToClipboard as utilCopyToClipboard } from "../utils/dataUtils";
import { stripMaskedSecrets } from "../utils/maskedSecretUtils";
import { truncateString } from "../utils/textUtils";
import AutoRouterConnectionTest from "./add_model/auto_router_connection_test";
import { AutoRouterTestTarget, buildAutoRouterTestTargets } from "./add_model/build_auto_router_test_targets";
import { normalizeTierModels } from "./add_model/complexity_router_tiers";
import {
  hasAutoRouterEditor,
  isAutoRouterDeployment,
  isComplexityRouter as isComplexityRouterParams,
} from "./add_model/auto_router_strategies";
import { canModifyModel } from "@/utils/modelPermissions";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import DeleteResourceModal from "./common_components/DeleteResourceModal";
import EditAutoRouterModal from "./edit_auto_router/edit_auto_router_modal";
import ReuseCredentialsModal from "./model_add/reuse_credentials";
import { toast } from "@/lib/toast";
import {
  CredentialItem,
  credentialCreateCall,
  credentialGetCall,
  credentialListCall,
  getGuardrailsList,
  modelDeleteCall,
  modelInfoV1Call,
  modelPatchUpdateCall,
  tagListCall,
  testConnectionRequest,
} from "./networking";
import { Logo } from "@/components/molecules/logo/Logo";
import UpdateModelCredentialsModal from "./update_model_credentials_modal";
import ModelInfoEditForm, { type ModelEditFormValues, type TouchedPricingField } from "./ModelInfoEditForm";
import { Tag } from "./tag_management/types";
import { getDisplayModelName } from "./view_model/model_name_display";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface ModelInfoViewProps {
  modelId: string;
  onClose: () => void;
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  onModelUpdate?: (updatedModel: any) => void;
  modelAccessGroups: string[] | null;
}

interface ComplexityRouterTierConfig {
  tiers?: {
    SIMPLE?: unknown;
    MEDIUM?: unknown;
    COMPLEX?: unknown;
    REASONING?: unknown;
  };
  semantic_keyword_matching?: boolean;
  embedding_model?: string;
  default_model?: string;
}

interface ComplexityRouterModelData {
  litellm_params?: {
    complexity_router_config?: ComplexityRouterTierConfig | string;
    complexity_router_default_model?: string;
  };
}

const buildComplexityRouterTestTargets = (
  modelData: ComplexityRouterModelData | null | undefined,
): AutoRouterTestTarget[] => {
  const rawConfig = modelData?.litellm_params?.complexity_router_config;
  let config: ComplexityRouterTierConfig = {};
  if (typeof rawConfig === "string") {
    try {
      config = JSON.parse(rawConfig);
    } catch {
      config = {};
    }
  } else if (rawConfig) {
    config = rawConfig;
  }

  const tiers: [string, string[]][] =
    config.tiers && typeof config.tiers === "object"
      ? Object.entries(config.tiers).map(([tier, models]) => [tier, normalizeTierModels(models)])
      : [];

  // Mirrors init_complexity_router_deployment (litellm/router.py): litellm_params wins, otherwise
  // pure tier-derivation. complexity_router_config.default_model is a UI-only marker the backend
  // never reads — folding it in here could point Test Connection at a model the router never
  // calls (see PR #36615 discussion).
  const effectiveDefaultModel = modelData?.litellm_params?.complexity_router_default_model || undefined;

  const testTargetParams = {
    tiers,
    semanticMatchingEnabled: Boolean(config.semantic_keyword_matching),
    embeddingModel: config.embedding_model,
    defaultModel: effectiveDefaultModel,
  };
  return buildAutoRouterTestTargets(testTargetParams);
};

export default function ModelInfoView({
  modelId,
  onClose,
  accessToken,
  userID,
  userRole,
  onModelUpdate,
  modelAccessGroups,
}: ModelInfoViewProps) {
  const queryClient = useQueryClient();
  const [localModelData, setLocalModelData] = useState<any>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [isCredentialModalOpen, setIsCredentialModalOpen] = useState(false);
  const [isUpdateCredentialsModalOpen, setIsUpdateCredentialsModalOpen] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [existingCredential, setExistingCredential] = useState<CredentialItem | null>(null);
  const [showCacheControl, setShowCacheControl] = useState(false);
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});
  const [isAutoRouterModalOpen, setIsAutoRouterModalOpen] = useState(false);
  const [isAutoRouterTestModalOpen, setIsAutoRouterTestModalOpen] = useState(false);
  const [autoRouterTestId, setAutoRouterTestId] = useState(0);
  const [autoRouterTestTargets, setAutoRouterTestTargets] = useState<AutoRouterTestTarget[]>([]);
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);
  const [tagsList, setTagsList] = useState<Record<string, Tag>>({});
  const [credentialsList, setCredentialsList] = useState<CredentialItem[]>([]);

  // Fetch model data using hook
  const { data: rawModelDataResponse, isLoading: isLoadingModel } = useModelsInfo(1, 50, undefined, modelId);
  const { data: modelCostMapData } = useModelCostMap();
  const { data: modelHubData } = useModelHub();
  const { data: teams } = useTeams();
  const ptuCostAttributionEnabled = usePtuCostAttributionEnabled();

  // Transform the model data
  const getProviderFromModel = (model: string) => {
    if (modelCostMapData !== null && modelCostMapData !== undefined) {
      if (typeof modelCostMapData == "object" && model in modelCostMapData) {
        return modelCostMapData[model]["litellm_provider"];
      }
    }
    return "openai";
  };

  const transformedModelData = useMemo(() => {
    if (!rawModelDataResponse?.data || rawModelDataResponse.data.length === 0) {
      return null;
    }
    const transformed = transformModelData(rawModelDataResponse, getProviderFromModel);
    return transformed.data[0] || null;
  }, [rawModelDataResponse, modelCostMapData]);

  // Keep modelData variable name for backwards compatibility
  const modelData = transformedModelData;

  const canEditModel = canModifyModel({ userRole, userID }, teams ?? null, {
    teamId: modelData?.model_info?.team_id,
    isDbModel: modelData?.model_info?.db_model === true,
  });
  const isAdmin = userRole === "Admin";
  // Editor-aware on purpose: an adaptive or quality router must not offer Edit Auto Router.
  const isAutoRouterModel = hasAutoRouterEditor(modelData?.litellm_params);
  // Broader than the editor check: adaptive and quality routers equally have no upstream
  // credential, so the credential actions are meaningless for every auto-router strategy.
  const isAnyAutoRouter = isAutoRouterDeployment(modelData?.litellm_params);
  const deleteLabel = isAnyAutoRouter ? "Delete Auto-Router" : "Delete Model";
  const isComplexityRouterModel = isComplexityRouterParams(modelData?.litellm_params);

  const usingExistingCredential =
    modelData?.litellm_params?.litellm_credential_name != null &&
    modelData?.litellm_params?.litellm_credential_name != undefined;

  // Initialize localModelData from modelData when available
  useEffect(() => {
    if (modelData && !localModelData) {
      let processedModelData = modelData;
      if (!processedModelData.litellm_model_name) {
        processedModelData = {
          ...processedModelData,
          litellm_model_name:
            processedModelData?.litellm_params?.litellm_model_name ??
            processedModelData?.litellm_params?.model ??
            processedModelData?.model_info?.key ??
            null,
        };
      }
      setLocalModelData(processedModelData);

      // Check if cache control is enabled
      if (processedModelData?.litellm_params?.cache_control_injection_points) {
        setShowCacheControl(true);
      }
    }
  }, [modelData, localModelData]);

  useEffect(() => {
    const getExistingCredential = async () => {
      if (!accessToken) return;
      if (usingExistingCredential) return;
      let existingCredentialResponse = await credentialGetCall(accessToken, null, modelId);
      setExistingCredential({
        credential_name: existingCredentialResponse["credential_name"],
        credential_values: existingCredentialResponse["credential_values"],
        credential_info: existingCredentialResponse["credential_info"],
      });
    };

    const getModelInfo = async () => {
      if (!accessToken) return;
      // Only fetch if we don't have modelData yet
      if (modelData) return;
      let modelInfoResponse = await modelInfoV1Call(accessToken, modelId);
      let specificModelData = modelInfoResponse.data[0];
      if (specificModelData && !specificModelData.litellm_model_name) {
        specificModelData = {
          ...specificModelData,
          litellm_model_name:
            specificModelData?.litellm_params?.litellm_model_name ??
            specificModelData?.litellm_params?.model ??
            specificModelData?.model_info?.key ??
            null,
        };
      }
      setLocalModelData(specificModelData);

      // Check if cache control is enabled
      if (specificModelData?.litellm_params?.cache_control_injection_points) {
        setShowCacheControl(true);
      }
    };

    const fetchGuardrails = async () => {
      if (!accessToken) return;
      try {
        const response = await getGuardrailsList(accessToken);
        const guardrailNames = response.guardrails.map((g: { guardrail_name: string }) => g.guardrail_name);
        setGuardrailsList(guardrailNames);
      } catch (error) {
        console.error("Failed to fetch guardrails:", error);
      }
    };

    const fetchTags = async () => {
      if (!accessToken) return;
      try {
        const response = await tagListCall(accessToken);
        setTagsList(response);
      } catch (error) {
        console.error("Failed to fetch tags:", error);
      }
    };

    const fetchCredentials = async () => {
      if (!accessToken) return;
      try {
        const response = await credentialListCall(accessToken);
        setCredentialsList(response.credentials || []);
      } catch (error) {
        console.error("Failed to fetch credentials:", error);
      }
    };

    getExistingCredential();
    getModelInfo();
    fetchGuardrails();
    fetchTags();
    fetchCredentials();
  }, [accessToken, modelId]);

  const handleReuseCredential = async (values: any) => {
    if (!accessToken) return;
    let credentialItem = {
      credential_name: values.credential_name,
      model_id: modelId,
      credential_info: {
        custom_llm_provider: localModelData.litellm_params?.custom_llm_provider,
      },
    };
    toast.info("Storing credential..");
    let credentialResponse = await credentialCreateCall(accessToken, credentialItem);
    toast.success("Credential stored successfully");
  };

  const handleModelUpdate = async (
    values: ModelEditFormValues,
    isFieldTouched: (field: TouchedPricingField) => boolean,
  ) => {
    try {
      if (!accessToken) return;
      setIsSaving(true);

      // Parse LiteLLM extra params from JSON text area
      let parsedExtraParams: Record<string, any> = {};
      try {
        parsedExtraParams = values.litellm_extra_params ? JSON.parse(values.litellm_extra_params) : {};
        delete parsedExtraParams.litellm_credential_name;
      } catch (e) {
        toast.fromError("Invalid JSON in LiteLLM Params");
        setIsSaving(false);
        return;
      }

      let updatedLitellmParams: Record<string, any> = {
        ...parsedExtraParams,
        model: values.litellm_model_name,
        api_base: values.api_base,
        custom_llm_provider: values.custom_llm_provider,
        organization: values.organization,
        tpm: values.tpm,
        rpm: values.rpm,
        max_retries: values.max_retries,
        timeout: values.timeout,
        stream_timeout: values.stream_timeout,
        tags: values.tags,
      };

      if (isFieldTouched("input_cost")) {
        if (values.input_cost !== undefined && values.input_cost !== null && values.input_cost !== "") {
          updatedLitellmParams.input_cost_per_token = Number(values.input_cost) / 1_000_000;
        } else {
          // Explicit null signals the backend to remove the pricing override.
          updatedLitellmParams.input_cost_per_token = null;
        }
      }
      if (isFieldTouched("output_cost")) {
        if (values.output_cost !== undefined && values.output_cost !== null && values.output_cost !== "") {
          updatedLitellmParams.output_cost_per_token = Number(values.output_cost) / 1_000_000;
        } else {
          updatedLitellmParams.output_cost_per_token = null;
        }
      }

      // Cache Read Cost:
      //   - explicit value provided → use it
      //   - field touched but empty → explicit null (signals backend to remove override)
      //   - only input_cost touched → fall back to input_cost (guarded against null)
      if (isFieldTouched("cache_read_cost") || isFieldTouched("input_cost")) {
        if (values.cache_read_cost !== undefined && values.cache_read_cost !== null && values.cache_read_cost !== "") {
          updatedLitellmParams.cache_read_input_token_cost = Number(values.cache_read_cost) / 1_000_000;
        } else if (isFieldTouched("cache_read_cost")) {
          updatedLitellmParams.cache_read_input_token_cost = null;
        } else if (
          updatedLitellmParams.input_cost_per_token !== undefined &&
          updatedLitellmParams.input_cost_per_token !== null
        ) {
          updatedLitellmParams.cache_read_input_token_cost = updatedLitellmParams.input_cost_per_token;
        }
      }

      // Cache Write Cost: explicit value if provided, else explicit null so the
      // backend removes the override and falls back to the model-level default.
      // Sending 0 here would persist a zero rate even when the user intended to unset it.
      if (isFieldTouched("cache_write_cost")) {
        if (
          values.cache_write_cost !== undefined &&
          values.cache_write_cost !== null &&
          values.cache_write_cost !== ""
        ) {
          updatedLitellmParams.cache_creation_input_token_cost = Number(values.cache_write_cost) / 1_000_000;
        } else {
          updatedLitellmParams.cache_creation_input_token_cost = null;
        }
      }

      if (values.litellm_credential_name) {
        updatedLitellmParams.litellm_credential_name = values.litellm_credential_name;
      } else {
        delete updatedLitellmParams.litellm_credential_name;
      }
      if (values.guardrails) {
        updatedLitellmParams.guardrails = values.guardrails;
      }
      if ((values.vector_store_ids?.length ?? 0) > 0) {
        updatedLitellmParams.vector_store_ids = values.vector_store_ids;
      } else if (values.vector_store_ids !== undefined) {
        // User explicitly cleared previously-set vector stores — send [] to clear on backend
        updatedLitellmParams.vector_store_ids = [];
      } else {
        delete updatedLitellmParams.vector_store_ids;
      }

      // Handle cache control settings
      if (values.cache_control && (values.cache_control_injection_points?.length ?? 0) > 0) {
        updatedLitellmParams.cache_control_injection_points = values.cache_control_injection_points;
      } else {
        delete updatedLitellmParams.cache_control_injection_points;
      }

      // Parse the model_info from the form values
      let updatedModelInfo;
      try {
        updatedModelInfo = values.model_info ? JSON.parse(values.model_info) : modelData.model_info;
        // Update access_groups from the form
        if (values.model_access_group) {
          updatedModelInfo = {
            ...updatedModelInfo,
            access_groups: values.model_access_group,
          };
        }
        // Override health_check_model from the form
        if (values.health_check_model !== undefined) {
          updatedModelInfo = {
            ...updatedModelInfo,
            health_check_model: values.health_check_model,
          };
        }
        updatedModelInfo = applyPtuModelInfo(updatedModelInfo, values, ptuCostAttributionEnabled);
      } catch (e) {
        toast.fromError("Invalid JSON in Model Info");
        return;
      }

      // Final guard: never PATCH a redacted secret. The /model/info snapshot that
      // seeds this form masks secrets, and any save re-sends the whole params blob;
      // without this strip a masked value would be re-encrypted over the real secret.
      // Credential rotation has its own dedicated path (UpdateModelCredentialsModal).
      const safeLitellmParams = stripMaskedSecrets(updatedLitellmParams);

      const updateData = {
        model_name: values.model_name,
        litellm_params: safeLitellmParams,
        model_info: updatedModelInfo,
      };

      await modelPatchUpdateCall(accessToken, updateData, modelId);

      const updatedModelData = {
        ...localModelData,
        model_name: values.model_name,
        litellm_model_name: values.litellm_model_name,
        litellm_params: safeLitellmParams,
        model_info: updatedModelInfo,
      };

      setLocalModelData(updatedModelData);

      if (onModelUpdate) {
        onModelUpdate(updatedModelData);
      }

      toast.success("Model settings updated successfully");
      setIsEditing(false);
    } catch (error) {
      console.error("Error updating model:", error);
      toast.fromError("Failed to update model settings");
    } finally {
      setIsSaving(false);
    }
  };

  // Show loading state
  if (isLoadingModel) {
    return (
      <div className="p-4">
        <Button variant="ghost" onClick={onClose} className="mb-4">
          <ArrowLeft className="size-4" />
          Back to Models
        </Button>
        <p className="text-sm">Loading...</p>
      </div>
    );
  }

  // Show not found if model is not found
  if (!modelData) {
    return (
      <div className="p-4">
        <Button variant="ghost" onClick={onClose} className="mb-4">
          <ArrowLeft className="size-4" />
          Back to Models
        </Button>
        <p className="text-sm">Model not found</p>
      </div>
    );
  }

  const handleTestConnection = async () => {
    if (!accessToken) return;
    if (isComplexityRouterModel) {
      const targets = buildComplexityRouterTestTargets(localModelData ?? modelData);
      if (targets.length === 0) {
        toast.warning("No complexity tiers are configured yet, so there is nothing to test.");
        return;
      }
      setAutoRouterTestTargets(targets);
      setAutoRouterTestId((id) => id + 1);
      setIsAutoRouterTestModalOpen(true);
      return;
    }
    try {
      toast.info("Testing connection...");
      const response = await testConnectionRequest(
        accessToken,
        {
          custom_llm_provider: localModelData.litellm_params.custom_llm_provider,
          litellm_credential_name: localModelData.litellm_params.litellm_credential_name,
          model: localModelData.litellm_model_name,
        },
        {
          // `id` is required to disambiguate when multiple deployments
          // share the same model_name (e.g. wildcard `openai/*` with two
          // different `api_base` values for failover). Without it the
          // backend silently falls back to deployments[0] and probes
          // the wrong endpoint.
          id: localModelData.model_info?.id,
          mode: localModelData.model_info?.mode,
        },
        localModelData.model_info?.mode,
      );

      if (response.status === "success") {
        toast.success("Connection test successful!");
      } else {
        throw new Error(response?.result?.error || response?.message || "Unknown error");
      }
    } catch (error) {
      if (error instanceof Error) {
        toast.error("Error testing connection: " + truncateString(error.message, 100));
      } else {
        toast.error("Error testing connection: " + String(error));
      }
    }
  };

  const handleDelete = async () => {
    try {
      setDeleteLoading(true);
      if (!accessToken) return;
      await modelDeleteCall(accessToken, modelId);
      toast.success("Model deleted successfully");

      if (onModelUpdate) {
        onModelUpdate({
          deleted: true,
          model_info: { id: modelId },
        });
      }

      onClose();
    } catch (error) {
      console.error("Error deleting the model:", error);
      toast.fromError("Failed to delete model");
    } finally {
      setDeleteLoading(false);
      setIsDeleteModalOpen(false);
    }
  };

  const copyToClipboard = async (text: string, key: string) => {
    const success = await utilCopyToClipboard(text);
    if (success) {
      setCopiedStates((prev) => ({ ...prev, [key]: true }));
      setTimeout(() => {
        setCopiedStates((prev) => ({ ...prev, [key]: false }));
      }, 2000);
    }
  };

  const handleAutoRouterUpdate = (updatedModel: any) => {
    setLocalModelData(updatedModel);
    if (onModelUpdate) {
      onModelUpdate(updatedModel);
    }
  };
  const isWildcardModel = modelData.litellm_model_name.includes("*");
  const wildcardProvider = modelData.litellm_model_name.split("/")[0];
  const healthCheckModelOptions =
    modelHubData?.data
      ?.filter(
        (model: any) =>
          model.providers?.includes(wildcardProvider) && model.model_group !== modelData.litellm_model_name,
      )
      .map((model: any) => ({ value: model.model_group, label: model.model_group })) || [];

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button variant="ghost" onClick={onClose} className="mb-4">
            <ArrowLeft className="size-4" />
            Back to Models
          </Button>
          <h2 className="text-xl font-semibold">Public Model Name: {getDisplayModelName(modelData)}</h2>
          <div className="flex items-center cursor-pointer">
            <span className="text-sm text-muted-foreground font-mono">{modelData.model_info.id}</span>
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label="Copy model ID"
              onClick={() => copyToClipboard(modelData.model_info.id, "model-id")}
              className={`left-2 z-raised transition-all duration-200 ${
                copiedStates["model-id"]
                  ? "text-success bg-success/10 border-success/20"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              }`}
            >
              {copiedStates["model-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
            </Button>
          </div>
        </div>
        <div className="flex gap-2">
          {(!isAnyAutoRouter || isComplexityRouterModel) && (
            <Button
              variant="outline"
              onClick={handleTestConnection}
              className="flex items-center gap-2"
              data-testid="test-connection-button"
            >
              <RefreshIcon className="h-4 w-4" />
              Test Connection
            </Button>
          )}

          {!isAnyAutoRouter && (
            <>
              <Button
                variant="outline"
                onClick={() => setIsUpdateCredentialsModalOpen(true)}
                className="flex items-center"
                disabled={!canEditModel}
                data-testid="update-api-key-button"
              >
                <KeyIcon className="h-4 w-4" />
                Update API Key
              </Button>

              <Button
                variant="outline"
                onClick={() => setIsCredentialModalOpen(true)}
                className="flex items-center"
                disabled={!isAdmin}
                data-testid="reuse-credentials-button"
              >
                <KeyIcon className="h-4 w-4" />
                Re-use Credentials
              </Button>
            </>
          )}
          <Button
            variant="destructive"
            onClick={() => setIsDeleteModalOpen(true)}
            className="flex items-center"
            disabled={!canEditModel}
            data-testid="delete-model-button"
          >
            <TrashIcon className="h-4 w-4" />
            {deleteLabel}
          </Button>
        </div>
      </div>

      <Tabs defaultValue="overview">
        <TabsList variant="line" className="mb-6 h-auto w-full justify-start rounded-none border-b p-0">
          <TabsTrigger value="overview" className="flex-none rounded-none px-4 py-2">
            Overview
          </TabsTrigger>
          <TabsTrigger value="raw" className="flex-none rounded-none px-4 py-2">
            Raw JSON
          </TabsTrigger>
        </TabsList>

        <div>
          <TabsContent value="overview" keepMounted>
            {/* Overview Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
              <Card className="block p-6">
                <p className="text-sm">Provider</p>
                <div className="mt-2 flex items-center space-x-2">
                  {modelData.provider && <Logo provider={modelData.provider} className="w-4 h-4" />}
                  <h3 className="text-lg font-medium">{modelData.provider || "Not Set"}</h3>
                </div>
              </Card>
              <Card className="block p-6">
                <p className="text-sm">LiteLLM Model</p>
                <div className="mt-2 overflow-hidden">
                  <SimpleTooltip content={modelData.litellm_model_name || "Not Set"} className="w-full min-w-0">
                    <div className="break-all text-sm font-medium leading-relaxed cursor-pointer">
                      {modelData.litellm_model_name || "Not Set"}
                    </div>
                  </SimpleTooltip>
                </div>
              </Card>
              <Card className="block p-6">
                <p className="text-sm">Pricing</p>
                <div className="mt-2">
                  <p className="text-sm">Input: ${modelData.input_cost}/1M tokens</p>
                  <p className="text-sm">Output: ${modelData.output_cost}/1M tokens</p>
                </div>
              </Card>
            </div>

            {/* Audit info shown as a subtle banner below the overview */}
            <div className="mb-6 text-sm text-muted-foreground flex items-center gap-x-6">
              <div className="flex items-center gap-x-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                Created At{" "}
                {modelData.model_info.created_at
                  ? new Date(modelData.model_info.created_at).toLocaleDateString("en-US", {
                      month: "short",
                      day: "numeric",
                      year: "numeric",
                    })
                  : "Not Set"}
              </div>
              <div className="flex items-center gap-x-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                  />
                </svg>
                Created By {modelData.model_info.created_by || "Not Set"}
              </div>
            </div>

            {/* Settings Card */}
            <Card className="block p-6">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-medium">Model Settings</h3>
                <div className="flex gap-2">
                  {isAutoRouterModel && canEditModel && !isEditing && (
                    <Button onClick={() => setIsAutoRouterModalOpen(true)} className="flex items-center">
                      Edit Auto Router
                    </Button>
                  )}
                  {canEditModel ? (
                    !isEditing && (
                      <Button onClick={() => setIsEditing(true)} className="flex items-center">
                        Edit Settings
                      </Button>
                    )
                  ) : (
                    <SimpleTooltip content="Only DB models can be edited. You must be an admin or the creator of the model to edit it.">
                      <Info className="size-4 text-muted-foreground" />
                    </SimpleTooltip>
                  )}
                </div>
              </div>
              {localModelData ? (
                <ModelInfoEditForm
                  localModelData={localModelData}
                  modelData={modelData}
                  accessToken={accessToken}
                  isEditing={isEditing}
                  isSaving={isSaving}
                  isWildcardModel={isWildcardModel}
                  ptuCostAttributionEnabled={ptuCostAttributionEnabled}
                  showCacheControl={showCacheControl}
                  setShowCacheControl={setShowCacheControl}
                  onCancel={() => setIsEditing(false)}
                  onSubmit={handleModelUpdate}
                  modelAccessGroups={modelAccessGroups}
                  guardrailsList={guardrailsList}
                  tagsList={tagsList}
                  credentialsList={credentialsList}
                  healthCheckModelOptions={healthCheckModelOptions}
                />
              ) : (
                <p className="text-sm">Loading...</p>
              )}
            </Card>
          </TabsContent>

          <TabsContent value="raw" keepMounted>
            <Card className="block p-6">
              <pre className="bg-muted p-4 rounded-sm text-xs overflow-auto">{JSON.stringify(modelData, null, 2)}</pre>
            </Card>
          </TabsContent>
        </div>
      </Tabs>

      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title={deleteLabel}
        alertMessage="This action cannot be undone."
        message={`Are you sure you want to delete this ${isAnyAutoRouter ? "auto-router" : "model"}?`}
        resourceInformationTitle="Model Information"
        resourceInformation={[
          {
            label: "Model Name",
            value: modelData?.model_name || "Not Set",
          },
          {
            label: "LiteLLM Model Name",
            value: modelData?.litellm_model_name || "Not Set",
          },
          {
            label: "Provider",
            value: modelData?.provider || "Not Set",
          },
          {
            label: "Created By",
            value: modelData?.model_info?.created_by || "Not Set",
          },
        ]}
        onCancel={() => setIsDeleteModalOpen(false)}
        onOk={handleDelete}
        confirmLoading={deleteLoading}
      />

      {isCredentialModalOpen && !usingExistingCredential ? (
        <ReuseCredentialsModal
          isVisible={isCredentialModalOpen}
          onCancel={() => setIsCredentialModalOpen(false)}
          onAddCredential={handleReuseCredential}
          existingCredential={existingCredential}
          setIsCredentialModalOpen={setIsCredentialModalOpen}
        />
      ) : (
        <Dialog open={isCredentialModalOpen} onOpenChange={(open) => !open && setIsCredentialModalOpen(false)}>
          <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Using Existing Credential</DialogTitle>
            </DialogHeader>
            <p className="text-sm">{modelData.litellm_params.litellm_credential_name}</p>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsCredentialModalOpen(false)}>
                Cancel
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}

      {isUpdateCredentialsModalOpen && accessToken && (
        <UpdateModelCredentialsModal
          open={isUpdateCredentialsModalOpen}
          onCancel={() => setIsUpdateCredentialsModalOpen(false)}
          accessToken={accessToken}
          modelId={modelId}
          onUpdated={() => {
            queryClient.invalidateQueries({ queryKey: ["models", "list"] });
          }}
        />
      )}

      {/* Edit Auto Router Modal */}
      <EditAutoRouterModal
        isVisible={isAutoRouterModalOpen}
        onCancel={() => setIsAutoRouterModalOpen(false)}
        onSuccess={handleAutoRouterUpdate}
        modelData={localModelData || modelData}
        accessToken={accessToken || ""}
        userRole={userRole || ""}
      />

      <Dialog open={isAutoRouterTestModalOpen} onOpenChange={(open) => !open && setIsAutoRouterTestModalOpen(false)}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[700px]">
          <DialogHeader>
            <DialogTitle>Connection Test Results</DialogTitle>
          </DialogHeader>
          {isAutoRouterTestModalOpen && accessToken && (
            <AutoRouterConnectionTest
              key={autoRouterTestId}
              accessToken={accessToken}
              targets={autoRouterTestTargets}
            />
          )}
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsAutoRouterTestModalOpen(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
