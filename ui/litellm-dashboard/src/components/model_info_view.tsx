import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { useModelHub, useModelsInfo } from "@/app/(dashboard)/hooks/models/useModels";
import { useQueryClient } from "@tanstack/react-query";
import { transformModelData } from "@/app/(dashboard)/models-and-endpoints/utils/modelDataTransformer";
import { InfoCircleOutlined } from "@ant-design/icons";
import { ArrowLeftIcon, KeyIcon, RefreshIcon, TrashIcon } from "@heroicons/react/outline";
import {
  Card,
  Grid,
  Tab,
  TabGroup,
  TabList,
  TabPanel,
  TabPanels,
  Text,
  TextInput,
  Title,
  Button as TremorButton,
} from "@tremor/react";
import { Button, Form, Input, Modal, Select, Tooltip } from "antd";
import VectorStoreSelector from "./vector_store_management/VectorStoreSelector";
import { CheckIcon, CopyIcon } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { copyToClipboard as utilCopyToClipboard } from "../utils/dataUtils";
import { isMaskedSecret, stripMaskedSecrets } from "../utils/maskedSecretUtils";
import { formItemValidateJSON, truncateString } from "../utils/textUtils";
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
import CacheControlSettings from "./add_model/cache_control_settings";
import DeleteResourceModal from "./common_components/DeleteResourceModal";
import EditAutoRouterModal from "./edit_auto_router/edit_auto_router_modal";
import ReuseCredentialsModal from "./model_add/reuse_credentials";
import NotificationsManager from "./molecules/notifications_manager";
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
import NumericalInput from "./shared/numerical_input";
import { Tag } from "./tag_management/types";
import { getDisplayModelName } from "./view_model/model_name_display";

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
}

interface ComplexityRouterModelData {
  litellm_params?: {
    complexity_router_config?: ComplexityRouterTierConfig | string;
    complexity_router_default_model?: string;
  };
}

const buildComplexityRouterTestTargets = (
  modelData: ComplexityRouterModelData | null | undefined,
  defaultTierLabel = "Default (unconfigured tiers)",
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

  const tierTargets = buildAutoRouterTestTargets({
    tiers: {
      SIMPLE: normalizeTierModels(config.tiers?.SIMPLE),
      MEDIUM: normalizeTierModels(config.tiers?.MEDIUM),
      COMPLEX: normalizeTierModels(config.tiers?.COMPLEX),
      REASONING: normalizeTierModels(config.tiers?.REASONING),
    },
    semanticMatchingEnabled: Boolean(config.semantic_keyword_matching),
    embeddingModel: config.embedding_model,
  });

  const defaultModel = modelData?.litellm_params?.complexity_router_default_model?.trim();
  if (!defaultModel || tierTargets.some((target) => target.modelGroup === defaultModel)) {
    return tierTargets;
  }
  return [...tierTargets, { labels: [defaultTierLabel], modelGroup: defaultModel, mode: "chat" as const }];
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
  const { t, i18n } = useTranslation("gateway");
  const [form] = Form.useForm();
  const queryClient = useQueryClient();
  const [localModelData, setLocalModelData] = useState<any>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [isCredentialModalOpen, setIsCredentialModalOpen] = useState(false);
  const [isUpdateCredentialsModalOpen, setIsUpdateCredentialsModalOpen] = useState(false);
  const [isDirty, setIsDirty] = useState(false);
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
  const deleteLabel = isAnyAutoRouter
    ? t("models.modelDetails.deleteAutoRouter")
    : t("models.modelDetails.deleteModel");
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
    NotificationsManager.info(t("models.modelDetails.storingCredential"));
    let credentialResponse = await credentialCreateCall(accessToken, credentialItem);
    NotificationsManager.success(t("models.modelDetails.credentialStored"));
  };

  const handleModelUpdate = async (values: any) => {
    try {
      if (!accessToken) return;
      setIsSaving(true);

      // Parse LiteLLM extra params from JSON text area
      let parsedExtraParams: Record<string, any> = {};
      try {
        parsedExtraParams = values.litellm_extra_params ? JSON.parse(values.litellm_extra_params) : {};
        delete parsedExtraParams.litellm_credential_name;
      } catch (e) {
        NotificationsManager.fromBackend(t("models.modelDetails.invalidLitellmParams"));
        setIsSaving(false);
        return;
      }

      let updatedLitellmParams = {
        ...values.litellm_params,
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

      if (form.isFieldTouched("input_cost")) {
        if (values.input_cost !== undefined && values.input_cost !== null && values.input_cost !== "") {
          updatedLitellmParams.input_cost_per_token = Number(values.input_cost) / 1_000_000;
        } else {
          // Explicit null signals the backend to remove the pricing override.
          updatedLitellmParams.input_cost_per_token = null;
        }
      }
      if (form.isFieldTouched("output_cost")) {
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
      if (form.isFieldTouched("cache_read_cost") || form.isFieldTouched("input_cost")) {
        if (values.cache_read_cost !== undefined && values.cache_read_cost !== null && values.cache_read_cost !== "") {
          updatedLitellmParams.cache_read_input_token_cost = Number(values.cache_read_cost) / 1_000_000;
        } else if (form.isFieldTouched("cache_read_cost")) {
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
      if (form.isFieldTouched("cache_write_cost")) {
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
      if (values.vector_store_ids?.length > 0) {
        updatedLitellmParams.vector_store_ids = values.vector_store_ids;
      } else if (values.vector_store_ids !== undefined) {
        // User explicitly cleared previously-set vector stores — send [] to clear on backend
        updatedLitellmParams.vector_store_ids = [];
      } else {
        delete updatedLitellmParams.vector_store_ids;
      }

      // Handle cache control settings
      if (values.cache_control && values.cache_control_injection_points?.length > 0) {
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
      } catch (e) {
        NotificationsManager.fromBackend(t("models.modelDetails.invalidModelInfo"));
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

      NotificationsManager.success(t("models.modelDetails.updated"));
      setIsDirty(false);
      setIsEditing(false);
    } catch (error) {
      console.error("Error updating model:", error);
      NotificationsManager.fromBackend(t("models.modelDetails.updateFailed"));
    } finally {
      setIsSaving(false);
    }
  };

  // Show loading state
  if (isLoadingModel) {
    return (
      <div className="p-4">
        <TremorButton icon={ArrowLeftIcon} variant="light" onClick={onClose} className="mb-4">
          {t("models.modelDetails.back")}
        </TremorButton>
        <Text>{t("models.modelDetails.loading")}</Text>
      </div>
    );
  }

  // Show not found if model is not found
  if (!modelData) {
    return (
      <div className="p-4">
        <TremorButton icon={ArrowLeftIcon} variant="light" onClick={onClose} className="mb-4">
          {t("models.modelDetails.back")}
        </TremorButton>
        <Text>{t("models.modelDetails.notFound")}</Text>
      </div>
    );
  }

  const handleTestConnection = async () => {
    if (!accessToken) return;
    if (isComplexityRouterModel) {
      const targets = buildComplexityRouterTestTargets(
        localModelData ?? modelData,
        t("models.modelDetails.defaultUnconfiguredTiers"),
      );
      if (targets.length === 0) {
        NotificationsManager.warning(t("models.modelDetails.noComplexityTiers"));
        return;
      }
      setAutoRouterTestTargets(targets);
      setAutoRouterTestId((id) => id + 1);
      setIsAutoRouterTestModalOpen(true);
      return;
    }
    try {
      NotificationsManager.info(t("models.modelDetails.testingConnection"));
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
        NotificationsManager.success(t("models.modelDetails.connectionSuccess"));
      } else {
        throw new Error(response?.result?.error || response?.message || t("models.modelDetails.unknownError"));
      }
    } catch (error) {
      if (error instanceof Error) {
        NotificationsManager.error(
          t("models.modelDetails.connectionError", { error: truncateString(error.message, 100) }),
        );
      } else {
        NotificationsManager.error(t("models.modelDetails.connectionError", { error: String(error) }));
      }
    }
  };

  const handleDelete = async () => {
    try {
      setDeleteLoading(true);
      if (!accessToken) return;
      await modelDeleteCall(accessToken, modelId);
      NotificationsManager.success(t("models.modelDetails.deleted"));

      if (onModelUpdate) {
        onModelUpdate({
          deleted: true,
          model_info: { id: modelId },
        });
      }

      onClose();
    } catch (error) {
      console.error("Error deleting the model:", error);
      NotificationsManager.fromBackend(t("models.modelDetails.deleteFailed"));
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

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <TremorButton icon={ArrowLeftIcon} variant="light" onClick={onClose} className="mb-4">
            {t("models.modelDetails.back")}
          </TremorButton>
          <Title>{t("models.modelDetails.publicName", { name: getDisplayModelName(modelData) })}</Title>
          <div className="flex items-center cursor-pointer">
            <Text className="text-gray-500 font-mono">{modelData.model_info.id}</Text>
            <Button
              type="text"
              size="small"
              icon={copiedStates["model-id"] ? <CheckIcon size={12} /> : <CopyIcon size={12} />}
              onClick={() => copyToClipboard(modelData.model_info.id, "model-id")}
              className={`left-2 z-10 transition-all duration-200 ${
                copiedStates["model-id"]
                  ? "text-green-600 bg-green-50 border-green-200"
                  : "text-gray-500 hover:text-gray-700 hover:bg-gray-100"
              }`}
            />
          </div>
        </div>
        <div className="flex gap-2">
          {(!isAnyAutoRouter || isComplexityRouterModel) && (
            <Button
              icon={<RefreshIcon className="h-4 w-4" />}
              onClick={handleTestConnection}
              className="flex items-center gap-2"
              data-testid="test-connection-button"
            >
              {t("models.modelDetails.testConnection")}
            </Button>
          )}

          {!isAnyAutoRouter && (
            <>
              <Button
                icon={<KeyIcon className="h-4 w-4" />}
                onClick={() => setIsUpdateCredentialsModalOpen(true)}
                className="flex items-center"
                disabled={!canEditModel}
                data-testid="update-api-key-button"
              >
                {t("models.modelDetails.updateApiKey")}
              </Button>

              <Button
                icon={<KeyIcon className="h-4 w-4" />}
                onClick={() => setIsCredentialModalOpen(true)}
                className="flex items-center"
                disabled={!isAdmin}
                data-testid="reuse-credentials-button"
              >
                {t("models.modelDetails.reuseCredentials")}
              </Button>
            </>
          )}
          <Button
            danger
            icon={<TrashIcon className="h-4 w-4" />}
            onClick={() => setIsDeleteModalOpen(true)}
            className="flex items-center"
            disabled={!canEditModel}
            data-testid="delete-model-button"
          >
            {deleteLabel}
          </Button>
        </div>
      </div>

      <TabGroup>
        <TabList className="mb-6">
          <Tab>{t("models.modelDetails.overview")}</Tab>
          <Tab>{t("models.modelDetails.rawJson")}</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            {/* Overview Grid */}
            <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6 mb-6">
              <Card>
                <Text>{t("models.modelDetails.provider")}</Text>
                <div className="mt-2 flex items-center space-x-2">
                  {modelData.provider && <Logo provider={modelData.provider} className="w-4 h-4" />}
                  <Title>{modelData.provider || t("models.modelDetails.notSet")}</Title>
                </div>
              </Card>
              <Card>
                <Text>{t("models.modelDetails.litellmModel")}</Text>
                <div className="mt-2 overflow-hidden">
                  <Tooltip title={modelData.litellm_model_name || t("models.modelDetails.notSet")}>
                    <div className="break-all text-sm font-medium leading-relaxed cursor-pointer">
                      {modelData.litellm_model_name || t("models.modelDetails.notSet")}
                    </div>
                  </Tooltip>
                </div>
              </Card>
              <Card>
                <Text>{t("models.modelDetails.pricing")}</Text>
                <div className="mt-2">
                  <Text>{t("models.modelDetails.inputPrice", { cost: modelData.input_cost })}</Text>
                  <Text>{t("models.modelDetails.outputPrice", { cost: modelData.output_cost })}</Text>
                </div>
              </Card>
            </Grid>

            {/* Audit info shown as a subtle banner below the overview */}
            <div className="mb-6 text-sm text-gray-500 flex items-center gap-x-6">
              <div className="flex items-center gap-x-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                {t("models.modelDetails.createdAt")}{" "}
                {modelData.model_info.created_at
                  ? new Date(modelData.model_info.created_at).toLocaleDateString(
                      i18n.resolvedLanguage === "ru" ? "ru-RU" : "en-US",
                      {
                        month: "short",
                        day: "numeric",
                        year: "numeric",
                      },
                    )
                  : t("models.modelDetails.notSet")}
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
                {t("models.modelDetails.createdBy")}{" "}
                {modelData.model_info.created_by || t("models.modelDetails.notSet")}
              </div>
            </div>

            {/* Settings Card */}
            <Card>
              <div className="flex justify-between items-center mb-4">
                <Title>{t("models.modelDetails.settings")}</Title>
                <div className="flex gap-2">
                  {isAutoRouterModel && canEditModel && !isEditing && (
                    <TremorButton onClick={() => setIsAutoRouterModalOpen(true)} className="flex items-center">
                      {t("models.modelDetails.editAutoRouter")}
                    </TremorButton>
                  )}
                  {canEditModel ? (
                    !isEditing && (
                      <TremorButton onClick={() => setIsEditing(true)} className="flex items-center">
                        {t("models.modelDetails.editSettings")}
                      </TremorButton>
                    )
                  ) : (
                    <Tooltip title={t("models.modelDetails.editDisabledTooltip")}>
                      <InfoCircleOutlined />
                    </Tooltip>
                  )}
                </div>
              </div>
              {localModelData ? (
                <Form
                  form={form}
                  onFinish={handleModelUpdate}
                  initialValues={{
                    model_name: localModelData.model_name,
                    litellm_model_name: localModelData.litellm_model_name,
                    api_base: localModelData.litellm_params.api_base,
                    custom_llm_provider: localModelData.litellm_params.custom_llm_provider,
                    organization: localModelData.litellm_params.organization,
                    tpm: localModelData.litellm_params.tpm,
                    rpm: localModelData.litellm_params.rpm,
                    max_retries: localModelData.litellm_params.max_retries,
                    timeout: localModelData.litellm_params.timeout,
                    stream_timeout: localModelData.litellm_params.stream_timeout,
                    input_cost: localModelData.litellm_params.input_cost_per_token
                      ? localModelData.litellm_params.input_cost_per_token * 1_000_000
                      : localModelData.model_info?.input_cost_per_token * 1_000_000 || null,
                    output_cost: localModelData.litellm_params?.output_cost_per_token
                      ? localModelData.litellm_params.output_cost_per_token * 1_000_000
                      : localModelData.model_info?.output_cost_per_token * 1_000_000 || null,
                    cache_read_cost:
                      localModelData.litellm_params?.cache_read_input_token_cost !== undefined &&
                      localModelData.litellm_params?.cache_read_input_token_cost !== null
                        ? localModelData.litellm_params.cache_read_input_token_cost * 1_000_000
                        : localModelData.model_info?.cache_read_input_token_cost !== undefined &&
                            localModelData.model_info?.cache_read_input_token_cost !== null
                          ? localModelData.model_info.cache_read_input_token_cost * 1_000_000
                          : null,
                    cache_write_cost:
                      localModelData.litellm_params?.cache_creation_input_token_cost !== undefined &&
                      localModelData.litellm_params?.cache_creation_input_token_cost !== null
                        ? localModelData.litellm_params.cache_creation_input_token_cost * 1_000_000
                        : localModelData.model_info?.cache_creation_input_token_cost !== undefined &&
                            localModelData.model_info?.cache_creation_input_token_cost !== null
                          ? localModelData.model_info.cache_creation_input_token_cost * 1_000_000
                          : null,
                    cache_control: localModelData.litellm_params?.cache_control_injection_points ? true : false,
                    cache_control_injection_points: localModelData.litellm_params?.cache_control_injection_points || [],
                    model_access_group: Array.isArray(localModelData.model_info?.access_groups)
                      ? localModelData.model_info.access_groups
                      : [],
                    guardrails: Array.isArray(localModelData.litellm_params?.guardrails)
                      ? localModelData.litellm_params.guardrails
                      : [],
                    vector_store_ids:
                      Array.isArray(localModelData.litellm_params?.vector_store_ids) &&
                      localModelData.litellm_params.vector_store_ids.length > 0
                        ? localModelData.litellm_params.vector_store_ids
                        : undefined,
                    tags: Array.isArray(localModelData.litellm_params?.tags) ? localModelData.litellm_params.tags : [],
                    health_check_model: isWildcardModel ? localModelData.model_info?.health_check_model : null,
                    litellm_credential_name: localModelData.litellm_params?.litellm_credential_name || "",
                    litellm_extra_params: JSON.stringify(
                      Object.fromEntries(
                        Object.entries(localModelData.litellm_params || {}).filter(
                          ([key, value]) => key !== "litellm_credential_name" && !isMaskedSecret(value),
                        ),
                      ),
                      null,
                      2,
                    ),
                  }}
                  layout="vertical"
                  onValuesChange={() => setIsDirty(true)}
                >
                  <div className="space-y-4">
                    <div className="space-y-4">
                      <div>
                        <Text className="font-medium">{t("models.modelDetails.modelName")}</Text>
                        {isEditing ? (
                          <Form.Item name="model_name" className="mb-0">
                            <TextInput placeholder={t("models.modelDetails.enterModelName")} />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">{localModelData.model_name}</div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">{t("models.modelDetails.litellmModelName")}</Text>
                        {isEditing ? (
                          <Form.Item name="litellm_model_name" className="mb-0">
                            <TextInput placeholder={t("models.modelDetails.enterLitellmModelName")} />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">{localModelData.litellm_model_name}</div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">{t("models.modelDetails.inputCost")}</Text>
                        {isEditing ? (
                          <Form.Item name="input_cost" className="mb-0">
                            <NumericalInput placeholder={t("models.modelDetails.enterInputCost")} />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData?.litellm_params?.input_cost_per_token
                              ? (localModelData.litellm_params?.input_cost_per_token * 1_000_000).toFixed(4)
                              : localModelData?.model_info?.input_cost_per_token
                                ? (localModelData.model_info.input_cost_per_token * 1_000_000).toFixed(4)
                                : t("models.modelDetails.notSet")}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">{t("models.modelDetails.outputCost")}</Text>
                        {isEditing ? (
                          <Form.Item name="output_cost" className="mb-0">
                            <NumericalInput placeholder={t("models.modelDetails.enterOutputCost")} />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData?.litellm_params?.output_cost_per_token
                              ? (localModelData.litellm_params.output_cost_per_token * 1_000_000).toFixed(4)
                              : localModelData?.model_info?.output_cost_per_token
                                ? (localModelData.model_info.output_cost_per_token * 1_000_000).toFixed(4)
                                : t("models.modelDetails.notSet")}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">{t("models.modelDetails.cacheReadCost")}</Text>
                        {isEditing ? (
                          <Form.Item
                            name="cache_read_cost"
                            className="mb-0"
                            tooltip={t("models.modelDetails.cacheReadTooltip")}
                          >
                            <NumericalInput placeholder={t("models.modelDetails.inputCostDefault")} />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData?.litellm_params?.cache_read_input_token_cost !== undefined &&
                            localModelData?.litellm_params?.cache_read_input_token_cost !== null
                              ? (localModelData.litellm_params.cache_read_input_token_cost * 1_000_000).toFixed(4)
                              : localModelData?.model_info?.cache_read_input_token_cost !== undefined &&
                                  localModelData?.model_info?.cache_read_input_token_cost !== null
                                ? (localModelData.model_info.cache_read_input_token_cost * 1_000_000).toFixed(4)
                                : t("models.modelDetails.notSet")}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">{t("models.modelDetails.cacheWriteCost")}</Text>
                        {isEditing ? (
                          <Form.Item
                            name="cache_write_cost"
                            className="mb-0"
                            tooltip={t("models.modelDetails.cacheWriteTooltip")}
                          >
                            <NumericalInput placeholder={t("models.modelDetails.inputCostDefault")} />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData?.litellm_params?.cache_creation_input_token_cost !== undefined &&
                            localModelData?.litellm_params?.cache_creation_input_token_cost !== null
                              ? (localModelData.litellm_params.cache_creation_input_token_cost * 1_000_000).toFixed(4)
                              : localModelData?.model_info?.cache_creation_input_token_cost !== undefined &&
                                  localModelData?.model_info?.cache_creation_input_token_cost !== null
                                ? (localModelData.model_info.cache_creation_input_token_cost * 1_000_000).toFixed(4)
                                : t("models.modelDetails.notSet")}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">{t("models.modelDetails.apiBase")}</Text>
                        {isEditing ? (
                          <Form.Item name="api_base" className="mb-0">
                            <TextInput placeholder={t("models.modelDetails.enterApiBase")} />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.api_base || t("models.modelDetails.notSet")}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">{t("models.modelDetails.customProvider")}</Text>
                        {isEditing ? (
                          <Form.Item name="custom_llm_provider" className="mb-0">
                            <TextInput placeholder={t("models.modelDetails.enterCustomProvider")} />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.custom_llm_provider || t("models.modelDetails.notSet")}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">{t("models.modelDetails.organization")}</Text>
                        {isEditing ? (
                          <Form.Item name="organization" className="mb-0">
                            <TextInput placeholder={t("models.modelDetails.enterOrganization")} />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.organization || t("models.modelDetails.notSet")}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">{t("models.modelDetails.tpm")}</Text>
                        {isEditing ? (
                          <Form.Item name="tpm" className="mb-0">
                            <NumericalInput placeholder={t("models.modelDetails.enterTpm")} />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.tpm || t("models.modelDetails.notSet")}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">{t("models.modelDetails.rpm")}</Text>
                        {isEditing ? (
                          <Form.Item name="rpm" className="mb-0">
                            <NumericalInput placeholder={t("models.modelDetails.enterRpm")} />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.rpm || t("models.modelDetails.notSet")}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">{t("models.modelDetails.maxRetries")}</Text>
                        {isEditing ? (
                          <Form.Item name="max_retries" className="mb-0">
                            <NumericalInput placeholder={t("models.modelDetails.enterMaxRetries")} />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.max_retries || t("models.modelDetails.notSet")}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">{t("models.modelDetails.timeout")}</Text>
                        {isEditing ? (
                          <Form.Item name="timeout" className="mb-0">
                            <NumericalInput placeholder={t("models.modelDetails.enterTimeout")} />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.timeout || t("models.modelDetails.notSet")}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">{t("models.modelDetails.streamTimeout")}</Text>
                        {isEditing ? (
                          <Form.Item name="stream_timeout" className="mb-0">
                            <NumericalInput placeholder={t("models.modelDetails.enterStreamTimeout")} />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.stream_timeout || t("models.modelDetails.notSet")}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">{t("models.modelDetails.accessGroups")}</Text>
                        {isEditing ? (
                          <Form.Item name="model_access_group" className="mb-0">
                            <Select
                              mode="tags"
                              showSearch
                              placeholder={t("models.modelDetails.selectAccessGroups")}
                              optionFilterProp="children"
                              tokenSeparators={[","]}
                              maxTagCount="responsive"
                              allowClear
                              style={{ width: "100%" }}
                              options={modelAccessGroups?.map((group) => ({
                                value: group,
                                label: group,
                              }))}
                            />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.model_info?.access_groups ? (
                              Array.isArray(localModelData.model_info.access_groups) ? (
                                localModelData.model_info.access_groups.length > 0 ? (
                                  <div className="flex flex-wrap gap-1">
                                    {localModelData.model_info.access_groups.map((group: string, index: number) => (
                                      <span
                                        key={index}
                                        className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                                      >
                                        {group}
                                      </span>
                                    ))}
                                  </div>
                                ) : (
                                  t("models.modelDetails.noGroups")
                                )
                              ) : (
                                localModelData.model_info.access_groups
                              )
                            ) : (
                              t("models.modelDetails.notSet")
                            )}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">
                          {t("models.modelDetails.guardrails")}
                          <Tooltip title={t("models.modelDetails.guardrailsTooltip")}>
                            <a
                              href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start"
                              target="_blank"
                              rel="noopener noreferrer"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                            </a>
                          </Tooltip>
                        </Text>
                        {isEditing ? (
                          <Form.Item name="guardrails" className="mb-0">
                            <Select
                              mode="tags"
                              showSearch
                              placeholder={t("models.modelDetails.selectGuardrails")}
                              optionFilterProp="children"
                              tokenSeparators={[","]}
                              maxTagCount="responsive"
                              allowClear
                              style={{ width: "100%" }}
                              options={guardrailsList.map((name) => ({
                                value: name,
                                label: name,
                              }))}
                            />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.guardrails ? (
                              Array.isArray(localModelData.litellm_params.guardrails) ? (
                                localModelData.litellm_params.guardrails.length > 0 ? (
                                  <div className="flex flex-wrap gap-1">
                                    {localModelData.litellm_params.guardrails.map(
                                      (guardrail: string, index: number) => (
                                        <span
                                          key={index}
                                          className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800"
                                        >
                                          {guardrail}
                                        </span>
                                      ),
                                    )}
                                  </div>
                                ) : (
                                  t("models.modelDetails.noGuardrails")
                                )
                              ) : (
                                localModelData.litellm_params.guardrails
                              )
                            ) : (
                              t("models.modelDetails.notSet")
                            )}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">
                          {t("models.modelDetails.knowledgeBases")}
                          <Tooltip title={t("models.modelDetails.knowledgeBasesTooltip")}>
                            <a
                              href="https://docs.litellm.ai/docs/completion/knowledgebase"
                              target="_blank"
                              rel="noopener noreferrer"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                            </a>
                          </Tooltip>
                        </Text>
                        {isEditing ? (
                          <Form.Item name="vector_store_ids" className="mb-0">
                            <VectorStoreSelector
                              onChange={() => {}}
                              accessToken={accessToken || ""}
                              placeholder={t("models.modelDetails.selectKnowledgeBases")}
                            />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.vector_store_ids ? (
                              Array.isArray(localModelData.litellm_params.vector_store_ids) ? (
                                localModelData.litellm_params.vector_store_ids.length > 0 ? (
                                  <div className="flex flex-wrap gap-1">
                                    {localModelData.litellm_params.vector_store_ids.map(
                                      (vsId: string, index: number) => (
                                        <span
                                          key={index}
                                          className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                                        >
                                          {vsId}
                                        </span>
                                      ),
                                    )}
                                  </div>
                                ) : (
                                  t("models.modelDetails.noKnowledgeBases")
                                )
                              ) : (
                                String(localModelData.litellm_params.vector_store_ids)
                              )
                            ) : (
                              t("models.modelDetails.notSet")
                            )}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">{t("models.modelDetails.tags")}</Text>
                        {isEditing ? (
                          <Form.Item name="tags" className="mb-0">
                            <Select
                              mode="tags"
                              showSearch
                              placeholder={t("models.modelDetails.selectTags")}
                              optionFilterProp="children"
                              tokenSeparators={[","]}
                              maxTagCount="responsive"
                              allowClear
                              style={{ width: "100%" }}
                              options={Object.values(tagsList).map((tag: Tag) => ({
                                value: tag.name,
                                label: tag.name,
                                title: tag.description || tag.name,
                              }))}
                            />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.tags ? (
                              Array.isArray(localModelData.litellm_params.tags) ? (
                                localModelData.litellm_params.tags.length > 0 ? (
                                  <div className="flex flex-wrap gap-1">
                                    {localModelData.litellm_params.tags.map((tag: string, index: number) => (
                                      <span
                                        key={index}
                                        className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800"
                                      >
                                        {tag}
                                      </span>
                                    ))}
                                  </div>
                                ) : (
                                  t("models.modelDetails.noTags")
                                )
                              ) : (
                                localModelData.litellm_params.tags
                              )
                            ) : (
                              t("models.modelDetails.notSet")
                            )}
                          </div>
                        )}
                      </div>
                      <div>
                        <Text className="font-medium">{t("models.modelDetails.existingCredentials")}</Text>
                        {isEditing ? (
                          <Form.Item name="litellm_credential_name" className="mb-0">
                            <Select
                              showSearch
                              placeholder={t("models.modelDetails.selectCredentials")}
                              optionFilterProp="children"
                              filterOption={(input, option) =>
                                (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
                              }
                              options={[
                                { value: "", label: t("models.modelDetails.none") },
                                ...credentialsList.map((credential) => ({
                                  value: credential.credential_name,
                                  label: credential.credential_name,
                                })),
                              ]}
                              allowClear
                            />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.litellm_credential_name || t("models.modelDetails.manual")}
                          </div>
                        )}
                      </div>

                      {isWildcardModel && (
                        <div>
                          <Text className="font-medium">{t("models.modelDetails.healthCheckModel")}</Text>
                          {isEditing ? (
                            <Form.Item name="health_check_model" className="mb-0">
                              <Select
                                showSearch
                                placeholder={t("models.modelDetails.selectHealthCheckModel")}
                                optionFilterProp="children"
                                allowClear
                                options={(() => {
                                  const wildcardProvider = modelData.litellm_model_name.split("/")[0];
                                  return (
                                    modelHubData?.data
                                      ?.filter((model: any) => {
                                        // Filter by provider to match the wildcard provider
                                        return (
                                          model.providers?.includes(wildcardProvider) &&
                                          model.model_group !== modelData.litellm_model_name
                                        );
                                      })
                                      .map((model: any) => ({
                                        value: model.model_group,
                                        label: model.model_group,
                                      })) || []
                                  );
                                })()}
                              />
                            </Form.Item>
                          ) : (
                            <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                              {localModelData.model_info?.health_check_model || t("models.modelDetails.notSet")}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Cache Control Section */}
                      {isEditing ? (
                        <CacheControlSettings
                          form={form}
                          showCacheControl={showCacheControl}
                          onCacheControlChange={(checked) => setShowCacheControl(checked)}
                        />
                      ) : (
                        <div>
                          <Text className="font-medium">{t("models.modelDetails.cacheControl")}</Text>
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.cache_control_injection_points ? (
                              <div>
                                <p>{t("models.modelDetails.enabled")}</p>
                                <div className="mt-2">
                                  {localModelData.litellm_params.cache_control_injection_points.map(
                                    (point: any, i: number) => (
                                      <div key={i} className="text-sm text-gray-600 mb-1">
                                        {t("models.modelDetails.location")}: {point.location},
                                        {point.role && (
                                          <span>
                                            {" "}
                                            {t("models.modelDetails.role")}: {point.role}
                                          </span>
                                        )}
                                        {point.index !== undefined && (
                                          <span>
                                            {" "}
                                            {t("models.modelDetails.index")}: {point.index}
                                          </span>
                                        )}
                                      </div>
                                    ),
                                  )}
                                </div>
                              </div>
                            ) : (
                              t("models.modelDetails.disabled")
                            )}
                          </div>
                        </div>
                      )}

                      <div>
                        <Text className="font-medium">{t("models.modelDetails.modelInfo")}</Text>
                        {isEditing ? (
                          <Form.Item name="model_info" className="mb-0">
                            <Input.TextArea
                              rows={4}
                              placeholder='{"gpt-4": 100, "claude-v1": 200}'
                              defaultValue={JSON.stringify(modelData.model_info, null, 2)}
                            />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            <pre className="bg-gray-100 p-2 rounded-sm text-xs overflow-auto mt-1">
                              {JSON.stringify(localModelData.model_info, null, 2)}
                            </pre>
                          </div>
                        )}
                      </div>
                      <div>
                        <Text className="font-medium">
                          {t("models.modelDetails.litellmParams")}
                          <Tooltip title={t("models.modelDetails.litellmParamsTooltip")}>
                            <a
                              href="https://docs.litellm.ai/docs/completion/input"
                              target="_blank"
                              rel="noopener noreferrer"
                              onClick={(e) => e.stopPropagation()}
                            >
                              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                            </a>
                          </Tooltip>
                        </Text>
                        {isEditing ? (
                          <Form.Item name="litellm_extra_params" rules={[{ validator: formItemValidateJSON }]}>
                            <Input.TextArea
                              rows={4}
                              placeholder='{
                  "rpm": 100,
                  "timeout": 0,
                  "stream_timeout": 0
                }'
                            />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            <pre className="bg-gray-100 p-2 rounded-sm text-xs overflow-auto mt-1">
                              {JSON.stringify(localModelData.litellm_params, null, 2)}
                            </pre>
                          </div>
                        )}
                      </div>
                      <div>
                        <Text className="font-medium">{t("models.modelDetails.teamId")}</Text>
                        <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                          {modelData.model_info.team_id || t("models.modelDetails.notSet")}
                        </div>
                      </div>
                    </div>

                    {isEditing && (
                      <div className="mt-6 flex justify-end gap-2">
                        <TremorButton
                          variant="secondary"
                          onClick={() => {
                            form.resetFields();
                            setIsDirty(false);
                            setIsEditing(false);
                          }}
                          disabled={isSaving}
                        >
                          {t("models.modelDetails.cancel")}
                        </TremorButton>
                        <TremorButton variant="primary" onClick={() => form.submit()} loading={isSaving}>
                          {t("models.modelDetails.saveChanges")}
                        </TremorButton>
                      </div>
                    )}
                  </div>
                </Form>
              ) : (
                <Text>{t("models.modelDetails.loading")}</Text>
              )}
            </Card>
          </TabPanel>

          <TabPanel>
            <Card>
              <pre className="bg-gray-100 p-4 rounded-sm text-xs overflow-auto">
                {JSON.stringify(modelData, null, 2)}
              </pre>
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>

      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title={deleteLabel}
        alertMessage={t("models.modelDetails.deleteWarning")}
        message={
          isAnyAutoRouter
            ? t("models.modelDetails.deleteAutoRouterMessage")
            : t("models.modelDetails.deleteModelMessage")
        }
        resourceInformationTitle={t("models.modelDetails.modelInformation")}
        resourceInformation={[
          {
            label: t("models.modelDetails.modelName"),
            value: modelData?.model_name || t("models.modelDetails.notSet"),
          },
          {
            label: t("models.modelDetails.litellmModelName"),
            value: modelData?.litellm_model_name || t("models.modelDetails.notSet"),
          },
          {
            label: t("models.modelDetails.provider"),
            value: modelData?.provider || t("models.modelDetails.notSet"),
          },
          {
            label: t("models.modelDetails.createdBy"),
            value: modelData?.model_info?.created_by || t("models.modelDetails.notSet"),
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
        <Modal
          open={isCredentialModalOpen}
          onCancel={() => setIsCredentialModalOpen(false)}
          title={t("models.modelDetails.usingExistingCredential")}
        >
          <Text>{modelData.litellm_params.litellm_credential_name}</Text>
        </Modal>
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

      <Modal
        title={t("models.modelDetails.connectionResults")}
        open={isAutoRouterTestModalOpen}
        onCancel={() => setIsAutoRouterTestModalOpen(false)}
        footer={[
          <Button key="close" onClick={() => setIsAutoRouterTestModalOpen(false)}>
            {t("models.modelDetails.close")}
          </Button>,
        ]}
        width={700}
      >
        {isAutoRouterTestModalOpen && accessToken && (
          <AutoRouterConnectionTest key={autoRouterTestId} accessToken={accessToken} targets={autoRouterTestTargets} />
        )}
      </Modal>
    </div>
  );
}
