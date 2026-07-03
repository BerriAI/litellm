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
import { copyToClipboard as utilCopyToClipboard } from "../utils/dataUtils";
import { formItemValidateJSON, truncateString } from "../utils/textUtils";
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
import { getProviderLogoAndName } from "./provider_info_helpers";
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

// The /model/info response redacts secrets by masking them (e.g. "sk-1****2345"),
// not by removing them. The edit form must never echo a masked value back on save:
// the backend would encrypt the asterisks and overwrite the real secret. A run of
// 2+ mask chars only appears in masker output (real config — incl. wildcard model
// names like "openai/*" — carries at most a single "*"), so this reliably detects a
// redacted value without a provider-metadata lookup. API-key rotation goes through
// UpdateModelCredentialsModal instead, which sends only the new key.
const isMaskedSecret = (value: unknown): boolean => {
  if (typeof value === "string") {
    return /\*{2,}/.test(value);
  }
  if (Array.isArray(value)) {
    return value.some(isMaskedSecret);
  }
  if (value !== null && typeof value === "object") {
    return Object.values(value as Record<string, unknown>).some(isMaskedSecret);
  }
  return false;
};

const stripMaskedSecrets = (
  params: Record<string, unknown>,
): { readonly safe: Record<string, unknown>; readonly dropped: readonly string[] } => {
  const entries = Object.entries(params);
  return {
    safe: Object.fromEntries(entries.filter(([, value]) => !isMaskedSecret(value))),
    dropped: entries.filter(([, value]) => isMaskedSecret(value)).map(([key]) => key),
  };
};

const seedTruthyCostPerMillion = (paramValue: unknown, modelInfoValue: unknown): number | null =>
  paramValue ? (paramValue as number) * 1_000_000 : (modelInfoValue as number) * 1_000_000 || null;

const seedExactCostPerMillion = (paramValue: unknown, modelInfoValue: unknown): number | null => {
  if (paramValue !== undefined && paramValue !== null) {
    return (paramValue as number) * 1_000_000;
  }
  if (modelInfoValue !== undefined && modelInfoValue !== null) {
    return (modelInfoValue as number) * 1_000_000;
  }
  return null;
};

type JsonRecordResult = { readonly ok: true; readonly value: Record<string, unknown> } | { readonly ok: false };

const parseJsonRecord = (raw: string | undefined): JsonRecordResult => {
  if (!raw) {
    return { ok: true, value: {} };
  }
  try {
    return { ok: true, value: JSON.parse(raw) as Record<string, unknown> };
  } catch {
    return { ok: false };
  }
};

const diffRecords = (current: Record<string, unknown>, initial: Record<string, unknown>): Record<string, unknown> =>
  Object.fromEntries(
    Object.entries(current).filter(([key, value]) => JSON.stringify(value) !== JSON.stringify(initial[key])),
  );

type ModelInfoPatch =
  | { readonly kind: "omit" }
  | { readonly kind: "invalid" }
  | { readonly kind: "include"; readonly value: Record<string, unknown> };

type ModelInfoPatchInput = {
  readonly changed: boolean;
  readonly modelInfoText: string | undefined;
  readonly accessGroups: unknown;
  readonly healthCheckModel: unknown;
  readonly baseModelInfo: Record<string, unknown>;
};

const buildModelInfoPatch = ({
  changed,
  modelInfoText,
  accessGroups,
  healthCheckModel,
  baseModelInfo,
}: ModelInfoPatchInput): ModelInfoPatch => {
  if (!changed) {
    return { kind: "omit" };
  }
  const parsed = parseJsonRecord(modelInfoText);
  if (!parsed.ok) {
    return { kind: "invalid" };
  }
  const base = modelInfoText ? parsed.value : baseModelInfo;
  return {
    kind: "include",
    value: {
      ...base,
      ...(accessGroups ? { access_groups: accessGroups } : {}),
      ...(healthCheckModel !== undefined ? { health_check_model: healthCheckModel } : {}),
      id: baseModelInfo.id,
      db_model: baseModelInfo.db_model,
    },
  };
};

const CHANGED_SCALAR_PARAM_FIELDS: ReadonlyArray<readonly [outboundKey: string, fieldName: string]> = [
  ["model", "litellm_model_name"],
  ["api_base", "api_base"],
  ["custom_llm_provider", "custom_llm_provider"],
  ["organization", "organization"],
  ["tpm", "tpm"],
  ["rpm", "rpm"],
  ["max_retries", "max_retries"],
  ["timeout", "timeout"],
  ["stream_timeout", "stream_timeout"],
  ["tags", "tags"],
];

export default function ModelInfoView({
  modelId,
  onClose,
  accessToken,
  userID,
  userRole,
  onModelUpdate,
  modelAccessGroups,
}: ModelInfoViewProps) {
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
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);
  const [tagsList, setTagsList] = useState<Record<string, Tag>>({});
  const [credentialsList, setCredentialsList] = useState<CredentialItem[]>([]);

  // Fetch model data using hook
  const { data: rawModelDataResponse, isLoading: isLoadingModel } = useModelsInfo(1, 50, undefined, modelId);
  const { data: modelCostMapData } = useModelCostMap();
  const { data: modelHubData } = useModelHub();

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

  const canEditModel =
    (userRole === "Admin" || modelData?.model_info?.created_by === userID) && modelData?.model_info?.db_model;
  const isAdmin = userRole === "Admin";
  const isAutoRouter = modelData?.litellm_params?.auto_router_config != null;

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
    NotificationsManager.info("Storing credential..");
    let credentialResponse = await credentialCreateCall(accessToken, credentialItem);
    NotificationsManager.success("Credential stored successfully");
  };

  const initialValues = useMemo<Record<string, unknown>>(() => {
    if (!localModelData) {
      return {};
    }
    const params = localModelData.litellm_params ?? {};
    const modelInfo = localModelData.model_info ?? {};
    const isWildcard =
      typeof localModelData.litellm_model_name === "string" && localModelData.litellm_model_name.includes("*");
    return {
      model_name: localModelData.model_name,
      litellm_model_name: localModelData.litellm_model_name,
      api_base: params.api_base,
      custom_llm_provider: params.custom_llm_provider,
      organization: params.organization,
      tpm: params.tpm,
      rpm: params.rpm,
      max_retries: params.max_retries,
      timeout: params.timeout,
      stream_timeout: params.stream_timeout,
      input_cost: seedTruthyCostPerMillion(params.input_cost_per_token, modelInfo.input_cost_per_token),
      output_cost: seedTruthyCostPerMillion(params.output_cost_per_token, modelInfo.output_cost_per_token),
      cache_read_cost: seedExactCostPerMillion(
        params.cache_read_input_token_cost,
        modelInfo.cache_read_input_token_cost,
      ),
      cache_write_cost: seedExactCostPerMillion(
        params.cache_creation_input_token_cost,
        modelInfo.cache_creation_input_token_cost,
      ),
      cache_control: params.cache_control_injection_points ? true : false,
      cache_control_injection_points: params.cache_control_injection_points || [],
      model_access_group: Array.isArray(modelInfo.access_groups) ? modelInfo.access_groups : [],
      guardrails: Array.isArray(params.guardrails) ? params.guardrails : [],
      vector_store_ids:
        Array.isArray(params.vector_store_ids) && params.vector_store_ids.length > 0
          ? params.vector_store_ids
          : undefined,
      tags: Array.isArray(params.tags) ? params.tags : [],
      health_check_model: isWildcard ? modelInfo.health_check_model : null,
      litellm_credential_name: params.litellm_credential_name || "",
      litellm_extra_params: JSON.stringify(
        Object.fromEntries(
          Object.entries(params).filter(([key, value]) => key !== "litellm_credential_name" && !isMaskedSecret(value)),
        ),
        null,
        2,
      ),
    };
  }, [localModelData]);

  const handleModelUpdate = async (values: any) => {
    try {
      if (!accessToken) return;
      setIsSaving(true);

      const parsedExtra = parseJsonRecord(values.litellm_extra_params);
      if (!parsedExtra.ok) {
        NotificationsManager.fromBackend("Invalid JSON in LiteLLM Params");
        setIsSaving(false);
        return;
      }
      const currentExtraParams = Object.fromEntries(
        Object.entries(parsedExtra.value).filter(([key]) => key !== "litellm_credential_name"),
      );
      const initialExtraRaw = initialValues.litellm_extra_params;
      const initialExtraResult = parseJsonRecord(typeof initialExtraRaw === "string" ? initialExtraRaw : undefined);
      const initialExtraParams = initialExtraResult.ok ? initialExtraResult.value : {};
      const changedExtraParams = diffRecords(currentExtraParams, initialExtraParams);

      const changedScalarParams = Object.fromEntries(
        CHANGED_SCALAR_PARAM_FIELDS.filter(([, fieldName]) => form.isFieldTouched(fieldName)).map(
          ([outboundKey, fieldName]) => [outboundKey, values[fieldName]],
        ),
      );

      const updatedLitellmParams: Record<string, unknown> = {
        ...changedExtraParams,
        ...changedScalarParams,
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

      if (form.isFieldTouched("litellm_credential_name") && values.litellm_credential_name) {
        updatedLitellmParams.litellm_credential_name = values.litellm_credential_name;
      }
      if (form.isFieldTouched("guardrails")) {
        updatedLitellmParams.guardrails = values.guardrails;
      }
      if (form.isFieldTouched("vector_store_ids")) {
        updatedLitellmParams.vector_store_ids = values.vector_store_ids?.length > 0 ? values.vector_store_ids : [];
      }

      // Handle cache control settings
      if (values.cache_control && values.cache_control_injection_points?.length > 0) {
        updatedLitellmParams.cache_control_injection_points = values.cache_control_injection_points;
      } else {
        delete updatedLitellmParams.cache_control_injection_points;
      }

      const modelInfoChanged =
        form.isFieldTouched("model_info") ||
        form.isFieldTouched("model_access_group") ||
        form.isFieldTouched("health_check_model");
      const modelInfoPatch = buildModelInfoPatch({
        changed: modelInfoChanged,
        modelInfoText: values.model_info,
        accessGroups: values.model_access_group,
        healthCheckModel: values.health_check_model,
        baseModelInfo: modelData.model_info,
      });
      if (modelInfoPatch.kind === "invalid") {
        NotificationsManager.fromBackend("Invalid JSON in Model Info");
        setIsSaving(false);
        return;
      }

      // Final guard: never PATCH a redacted secret. The /model/info snapshot that
      // seeds this form masks secrets; without this strip a masked value (top-level
      // or nested inside an object/array) would be re-encrypted over the real secret.
      // Credential rotation has its own dedicated path (UpdateModelCredentialsModal).
      const { safe: safeLitellmParams, dropped: droppedMaskedParams } = stripMaskedSecrets(updatedLitellmParams);
      if (droppedMaskedParams.length > 0) {
        NotificationsManager.warning(
          `These fields still held a redacted value and were not saved: ${droppedMaskedParams.join(
            ", ",
          )}. Re-enter their real value to change them, or rotate the API key from "Update API Key".`,
        );
      }

      const modelNameChanged = form.isFieldTouched("model_name");
      const updateData = {
        ...(modelNameChanged ? { model_name: values.model_name } : {}),
        litellm_params: safeLitellmParams,
        ...(modelInfoPatch.kind === "include" ? { model_info: modelInfoPatch.value } : {}),
      };

      await modelPatchUpdateCall(accessToken, updateData, modelId);

      const updatedModelData = {
        ...localModelData,
        ...(modelNameChanged ? { model_name: values.model_name } : {}),
        ...(form.isFieldTouched("litellm_model_name") ? { litellm_model_name: values.litellm_model_name } : {}),
        litellm_params: { ...localModelData.litellm_params, ...safeLitellmParams },
        ...(modelInfoPatch.kind === "include" ? { model_info: modelInfoPatch.value } : {}),
      };

      setLocalModelData(updatedModelData);

      if (onModelUpdate) {
        onModelUpdate(updatedModelData);
      }

      NotificationsManager.success("Model settings updated successfully");
      setIsDirty(false);
      setIsEditing(false);
    } catch (error) {
      console.error("Error updating model:", error);
      NotificationsManager.fromBackend("Failed to update model settings");
    } finally {
      setIsSaving(false);
    }
  };

  // Show loading state
  if (isLoadingModel) {
    return (
      <div className="p-4">
        <TremorButton icon={ArrowLeftIcon} variant="light" onClick={onClose} className="mb-4">
          Back to Models
        </TremorButton>
        <Text>Loading...</Text>
      </div>
    );
  }

  // Show not found if model is not found
  if (!modelData) {
    return (
      <div className="p-4">
        <TremorButton icon={ArrowLeftIcon} variant="light" onClick={onClose} className="mb-4">
          Back to Models
        </TremorButton>
        <Text>Model not found</Text>
      </div>
    );
  }

  const handleTestConnection = async () => {
    if (!accessToken) return;
    try {
      NotificationsManager.info("Testing connection...");
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
        NotificationsManager.success("Connection test successful!");
      } else {
        throw new Error(response?.result?.error || response?.message || "Unknown error");
      }
    } catch (error) {
      if (error instanceof Error) {
        NotificationsManager.error("Error testing connection: " + truncateString(error.message, 100));
      } else {
        NotificationsManager.error("Error testing connection: " + String(error));
      }
    }
  };

  const handleDelete = async () => {
    try {
      setDeleteLoading(true);
      if (!accessToken) return;
      await modelDeleteCall(accessToken, modelId);
      NotificationsManager.success("Model deleted successfully");

      if (onModelUpdate) {
        onModelUpdate({
          deleted: true,
          model_info: { id: modelId },
        });
      }

      onClose();
    } catch (error) {
      console.error("Error deleting the model:", error);
      NotificationsManager.fromBackend("Failed to delete model");
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
            Back to Models
          </TremorButton>
          <Title>Public Model Name: {getDisplayModelName(modelData)}</Title>
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
          <Button
            icon={<RefreshIcon className="h-4 w-4" />}
            onClick={handleTestConnection}
            className="flex items-center gap-2"
            data-testid="test-connection-button"
          >
            Test Connection
          </Button>

          <Button
            icon={<KeyIcon className="h-4 w-4" />}
            onClick={() => setIsUpdateCredentialsModalOpen(true)}
            className="flex items-center"
            disabled={!canEditModel}
            data-testid="update-api-key-button"
          >
            Update API Key
          </Button>

          <Button
            icon={<KeyIcon className="h-4 w-4" />}
            onClick={() => setIsCredentialModalOpen(true)}
            className="flex items-center"
            disabled={!isAdmin}
            data-testid="reuse-credentials-button"
          >
            Re-use Credentials
          </Button>
          <Button
            danger
            icon={<TrashIcon className="h-4 w-4" />}
            onClick={() => setIsDeleteModalOpen(true)}
            className="flex items-center"
            disabled={!canEditModel}
            data-testid="delete-model-button"
          >
            Delete Model
          </Button>
        </div>
      </div>

      <TabGroup>
        <TabList className="mb-6">
          <Tab>Overview</Tab>
          <Tab>Raw JSON</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            {/* Overview Grid */}
            <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6 mb-6">
              <Card>
                <Text>Provider</Text>
                <div className="mt-2 flex items-center space-x-2">
                  {modelData.provider && (
                    <img
                      src={getProviderLogoAndName(modelData.provider).logo}
                      alt={`${modelData.provider} logo`}
                      className="w-4 h-4"
                      onError={(e) => {
                        const target = e.currentTarget as HTMLImageElement;
                        const parent = target.parentElement;
                        if (!parent || !parent.contains(target)) {
                          return;
                        }

                        try {
                          const fallbackDiv = document.createElement("div");
                          fallbackDiv.className =
                            "w-4 h-4 rounded-full bg-gray-200 flex items-center justify-center text-xs";
                          fallbackDiv.textContent = modelData.provider?.charAt(0) || "-";
                          parent.replaceChild(fallbackDiv, target);
                        } catch (error) {
                          console.error("Failed to replace provider logo fallback:", error);
                        }
                      }}
                    />
                  )}
                  <Title>{modelData.provider || "Not Set"}</Title>
                </div>
              </Card>
              <Card>
                <Text>LiteLLM Model</Text>
                <div className="mt-2 overflow-hidden">
                  <Tooltip title={modelData.litellm_model_name || "Not Set"}>
                    <div className="break-all text-sm font-medium leading-relaxed cursor-pointer">
                      {modelData.litellm_model_name || "Not Set"}
                    </div>
                  </Tooltip>
                </div>
              </Card>
              <Card>
                <Text>Pricing</Text>
                <div className="mt-2">
                  <Text>Input: ${modelData.input_cost}/1M tokens</Text>
                  <Text>Output: ${modelData.output_cost}/1M tokens</Text>
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
            <Card>
              <div className="flex justify-between items-center mb-4">
                <Title>Model Settings</Title>
                <div className="flex gap-2">
                  {isAutoRouter && canEditModel && !isEditing && (
                    <TremorButton onClick={() => setIsAutoRouterModalOpen(true)} className="flex items-center">
                      Edit Auto Router
                    </TremorButton>
                  )}
                  {canEditModel ? (
                    !isEditing && (
                      <TremorButton onClick={() => setIsEditing(true)} className="flex items-center">
                        Edit Settings
                      </TremorButton>
                    )
                  ) : (
                    <Tooltip title="Only DB models can be edited. You must be an admin or the creator of the model to edit it.">
                      <InfoCircleOutlined />
                    </Tooltip>
                  )}
                </div>
              </div>
              {localModelData ? (
                <Form
                  form={form}
                  onFinish={handleModelUpdate}
                  initialValues={initialValues}
                  layout="vertical"
                  onValuesChange={() => setIsDirty(true)}
                >
                  <div className="space-y-4">
                    <div className="space-y-4">
                      <div>
                        <Text className="font-medium">Model Name</Text>
                        {isEditing ? (
                          <Form.Item name="model_name" className="mb-0">
                            <TextInput placeholder="Enter model name" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">{localModelData.model_name}</div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">LiteLLM Model Name</Text>
                        {isEditing ? (
                          <Form.Item name="litellm_model_name" className="mb-0">
                            <TextInput placeholder="Enter LiteLLM model name" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">{localModelData.litellm_model_name}</div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">Input Cost (per 1M tokens)</Text>
                        {isEditing ? (
                          <Form.Item name="input_cost" className="mb-0">
                            <NumericalInput placeholder="Enter input cost" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData?.litellm_params?.input_cost_per_token
                              ? (localModelData.litellm_params?.input_cost_per_token * 1_000_000).toFixed(4)
                              : localModelData?.model_info?.input_cost_per_token
                                ? (localModelData.model_info.input_cost_per_token * 1_000_000).toFixed(4)
                                : "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">Output Cost (per 1M tokens)</Text>
                        {isEditing ? (
                          <Form.Item name="output_cost" className="mb-0">
                            <NumericalInput placeholder="Enter output cost" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData?.litellm_params?.output_cost_per_token
                              ? (localModelData.litellm_params.output_cost_per_token * 1_000_000).toFixed(4)
                              : localModelData?.model_info?.output_cost_per_token
                                ? (localModelData.model_info.output_cost_per_token * 1_000_000).toFixed(4)
                                : "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">Cache Read Cost (per 1M tokens)</Text>
                        {isEditing ? (
                          <Form.Item
                            name="cache_read_cost"
                            className="mb-0"
                            tooltip="If left blank on save, defaults to Input Cost."
                          >
                            <NumericalInput placeholder="Defaults to Input Cost if blank" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData?.litellm_params?.cache_read_input_token_cost !== undefined &&
                            localModelData?.litellm_params?.cache_read_input_token_cost !== null
                              ? (localModelData.litellm_params.cache_read_input_token_cost * 1_000_000).toFixed(4)
                              : localModelData?.model_info?.cache_read_input_token_cost !== undefined &&
                                  localModelData?.model_info?.cache_read_input_token_cost !== null
                                ? (localModelData.model_info.cache_read_input_token_cost * 1_000_000).toFixed(4)
                                : "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">Cache Write Cost (per 1M tokens)</Text>
                        {isEditing ? (
                          <Form.Item
                            name="cache_write_cost"
                            className="mb-0"
                            tooltip="If left blank on save, defaults to Input Cost (backend falls back to input_cost_per_token)."
                          >
                            <NumericalInput placeholder="Defaults to Input Cost if blank" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData?.litellm_params?.cache_creation_input_token_cost !== undefined &&
                            localModelData?.litellm_params?.cache_creation_input_token_cost !== null
                              ? (localModelData.litellm_params.cache_creation_input_token_cost * 1_000_000).toFixed(4)
                              : localModelData?.model_info?.cache_creation_input_token_cost !== undefined &&
                                  localModelData?.model_info?.cache_creation_input_token_cost !== null
                                ? (localModelData.model_info.cache_creation_input_token_cost * 1_000_000).toFixed(4)
                                : "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">API Base</Text>
                        {isEditing ? (
                          <Form.Item name="api_base" className="mb-0">
                            <TextInput placeholder="Enter API base" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.api_base || "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">Custom LLM Provider</Text>
                        {isEditing ? (
                          <Form.Item name="custom_llm_provider" className="mb-0">
                            <TextInput placeholder="Enter custom LLM provider" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.custom_llm_provider || "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">Organization</Text>
                        {isEditing ? (
                          <Form.Item name="organization" className="mb-0">
                            <TextInput placeholder="Enter organization" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.organization || "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">TPM (Tokens per Minute)</Text>
                        {isEditing ? (
                          <Form.Item name="tpm" className="mb-0">
                            <NumericalInput placeholder="Enter TPM" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.tpm || "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">RPM (Requests per Minute)</Text>
                        {isEditing ? (
                          <Form.Item name="rpm" className="mb-0">
                            <NumericalInput placeholder="Enter RPM" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.rpm || "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">Max Retries</Text>
                        {isEditing ? (
                          <Form.Item name="max_retries" className="mb-0">
                            <NumericalInput placeholder="Enter max retries" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.max_retries || "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">Timeout (seconds)</Text>
                        {isEditing ? (
                          <Form.Item name="timeout" className="mb-0">
                            <NumericalInput placeholder="Enter timeout" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.timeout || "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">Stream Timeout (seconds)</Text>
                        {isEditing ? (
                          <Form.Item name="stream_timeout" className="mb-0">
                            <NumericalInput placeholder="Enter stream timeout" />
                          </Form.Item>
                        ) : (
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.stream_timeout || "Not Set"}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">Model Access Groups</Text>
                        {isEditing ? (
                          <Form.Item name="model_access_group" className="mb-0">
                            <Select
                              mode="tags"
                              showSearch
                              placeholder="Select existing groups or type to create new ones"
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
                                  "No groups assigned"
                                )
                              ) : (
                                localModelData.model_info.access_groups
                              )
                            ) : (
                              "Not Set"
                            )}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">
                          Guardrails
                          <Tooltip title="Apply safety guardrails to this model to filter content or enforce policies">
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
                              placeholder="Select existing guardrails or type to create new ones"
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
                                  "No guardrails assigned"
                                )
                              ) : (
                                localModelData.litellm_params.guardrails
                              )
                            ) : (
                              "Not Set"
                            )}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">
                          Attached Knowledge Bases (RAG)
                          <Tooltip title="Vector stores used for RAG. Every request to this model will automatically retrieve context from these knowledge bases.">
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
                              placeholder="Select knowledge bases (optional)"
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
                                  "No knowledge bases attached"
                                )
                              ) : (
                                String(localModelData.litellm_params.vector_store_ids)
                              )
                            ) : (
                              "Not Set"
                            )}
                          </div>
                        )}
                      </div>

                      <div>
                        <Text className="font-medium">Tags</Text>
                        {isEditing ? (
                          <Form.Item name="tags" className="mb-0">
                            <Select
                              mode="tags"
                              showSearch
                              placeholder="Select existing tags or type to create new ones"
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
                                  "No tags assigned"
                                )
                              ) : (
                                localModelData.litellm_params.tags
                              )
                            ) : (
                              "Not Set"
                            )}
                          </div>
                        )}
                      </div>
                      <div>
                        <Text className="font-medium">Existing Credentials</Text>
                        {isEditing ? (
                          <Form.Item name="litellm_credential_name" className="mb-0">
                            <Select
                              showSearch
                              placeholder="Select or search for existing credentials"
                              optionFilterProp="children"
                              filterOption={(input, option) =>
                                (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
                              }
                              options={[
                                { value: "", label: "None" },
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
                            {localModelData.litellm_params?.litellm_credential_name || "Manual"}
                          </div>
                        )}
                      </div>

                      {isWildcardModel && (
                        <div>
                          <Text className="font-medium">Health Check Model</Text>
                          {isEditing ? (
                            <Form.Item name="health_check_model" className="mb-0">
                              <Select
                                showSearch
                                placeholder="Select existing health check model"
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
                              {localModelData.model_info?.health_check_model || "Not Set"}
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
                          <Text className="font-medium">Cache Control</Text>
                          <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                            {localModelData.litellm_params?.cache_control_injection_points ? (
                              <div>
                                <p>Enabled</p>
                                <div className="mt-2">
                                  {localModelData.litellm_params.cache_control_injection_points.map(
                                    (point: any, i: number) => (
                                      <div key={i} className="text-sm text-gray-600 mb-1">
                                        Location: {point.location},{point.role && <span> Role: {point.role}</span>}
                                        {point.index !== undefined && <span> Index: {point.index}</span>}
                                      </div>
                                    ),
                                  )}
                                </div>
                              </div>
                            ) : (
                              "Disabled"
                            )}
                          </div>
                        </div>
                      )}

                      <div>
                        <Text className="font-medium">Model Info</Text>
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
                          LiteLLM Params
                          <Tooltip title="Optional litellm params used for making a litellm.completion() call. Some params are automatically added by LiteLLM.">
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
                        <Text className="font-medium">Team ID</Text>
                        <div className="mt-1 p-2 bg-gray-50 rounded-sm">
                          {modelData.model_info.team_id || "Not Set"}
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
                          Cancel
                        </TremorButton>
                        <TremorButton variant="primary" onClick={() => form.submit()} loading={isSaving}>
                          Save Changes
                        </TremorButton>
                      </div>
                    )}
                  </div>
                </Form>
              ) : (
                <Text>Loading...</Text>
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
        title="Delete Model"
        alertMessage="This action cannot be undone."
        message="Are you sure you want to delete this model?"
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
        <Modal
          open={isCredentialModalOpen}
          onCancel={() => setIsCredentialModalOpen(false)}
          title="Using Existing Credential"
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
    </div>
  );
}
