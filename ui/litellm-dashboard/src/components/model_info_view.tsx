import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { useModelHub, useModelsInfo } from "@/app/(dashboard)/hooks/models/useModels";
import { transformModelData } from "@/app/(dashboard)/models-and-endpoints/utils/modelDataTransformer";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ArrowLeft as ArrowLeftIcon,
  Check as CheckIcon,
  Copy as CopyIcon,
  Info as InfoCircleOutlined,
  Key as KeyIcon,
  RefreshCcw as RefreshIcon,
  Trash2 as TrashIcon,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import {
  Controller,
  FormProvider,
  useForm,
  useFormContext,
} from "react-hook-form";
import { copyToClipboard as utilCopyToClipboard } from "../utils/dataUtils";
import { truncateString } from "../utils/textUtils";
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
import NumericalInput from "./shared/numerical_input";
import { Tag } from "./tag_management/types";
import VectorStoreSelector from "./vector_store_management/VectorStoreSelector";
import { getDisplayModelName } from "./view_model/model_name_display";

interface ModelInfoViewProps {
  modelId: string;
  onClose: () => void;
  accessToken: string | null;
  userID: string | null;
  userRole: string | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onModelUpdate?: (updatedModel: any) => void;
  modelAccessGroups: string[] | null;
}

type ModelSettingsFormValues = {
  model_name: string;
  litellm_model_name: string;
  api_base: string;
  custom_llm_provider: string;
  organization: string;
  tpm: number | null;
  rpm: number | null;
  max_retries: number | null;
  timeout: number | null;
  stream_timeout: number | null;
  input_cost: number | null;
  output_cost: number | null;
  cache_control: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  cache_control_injection_points: any[];
  model_access_group: string[];
  guardrails: string[];
  vector_store_ids: string[] | undefined;
  tags: string[];
  health_check_model: string | null;
  litellm_credential_name: string;
  litellm_extra_params: string;
  model_info: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  litellm_params?: Record<string, any>;
};

/**
 * Shadcn-style multi-select chip input that also supports free-text
 * creation of new entries (mirrors antd's `Select mode="tags"` behavior).
 */
function TagsLikeMultiSelect({
  value,
  onChange,
  options,
  placeholder,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: { value: string; label: string; title?: string }[];
  placeholder?: string;
}) {
  const [inputValue, setInputValue] = useState("");

  const addValue = (v: string) => {
    const trimmed = v.trim();
    if (!trimmed) return;
    if (!value.includes(trimmed)) {
      onChange([...value, trimmed]);
    }
  };

  const remove = (v: string) => {
    onChange(value.filter((x) => x !== v));
  };

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-1 items-center border border-input rounded-md p-2 min-h-10 bg-background">
        {value.map((v) => (
          <Badge
            key={v}
            variant="secondary"
            className="flex items-center gap-1"
          >
            {options.find((o) => o.value === v)?.label ?? v}
            <button
              type="button"
              aria-label={`Remove ${v}`}
              onClick={() => remove(v)}
              className="ml-1 inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
            >
              ×
            </button>
          </Badge>
        ))}
        <input
          type="text"
          placeholder={value.length === 0 ? placeholder : undefined}
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === ",") {
              e.preventDefault();
              addValue(inputValue);
              setInputValue("");
            } else if (
              e.key === "Backspace" &&
              inputValue === "" &&
              value.length > 0
            ) {
              remove(value[value.length - 1]);
            }
          }}
          onBlur={() => {
            if (inputValue) {
              addValue(inputValue);
              setInputValue("");
            }
          }}
          className="flex-1 min-w-[100px] outline-none bg-transparent text-sm"
        />
      </div>
      {options.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {options
            .filter((o) => !value.includes(o.value))
            .slice(0, 10)
            .map((o) => (
              <button
                type="button"
                key={o.value}
                title={o.title}
                onClick={() => addValue(o.value)}
                className="text-xs px-2 py-0.5 rounded-md border border-border text-muted-foreground hover:bg-muted"
              >
                + {o.label}
              </button>
            ))}
        </div>
      )}
    </div>
  );
}

/**
 * Shadcn-style single-select with search + clear (mirrors antd
 * `Select showSearch allowClear`).
 */
function SearchableSingleSelect({
  value,
  onChange,
  options,
  placeholder,
}: {
  value: string;
  onChange: (next: string) => void;
  options: { value: string; label: string }[];
  placeholder?: string;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const filtered = options.filter((o) =>
    o.label.toLowerCase().includes(query.toLowerCase()),
  );
  const selected = options.find((o) => o.value === value);
  return (
    <div className="relative">
      <div className="flex items-center border border-input rounded-md bg-background">
        <input
          type="text"
          className="flex-1 px-3 py-2 text-sm bg-transparent outline-none"
          placeholder={placeholder}
          value={open ? query : selected?.label ?? value ?? ""}
          onFocus={() => {
            setOpen(true);
            setQuery("");
          }}
          onBlur={() => setTimeout(() => setOpen(false), 150)}
          onChange={(e) => {
            setQuery(e.target.value);
            setOpen(true);
          }}
        />
        {value && (
          <button
            type="button"
            aria-label="Clear selection"
            onClick={() => onChange("")}
            className="px-2 text-muted-foreground hover:text-foreground"
          >
            ×
          </button>
        )}
      </div>
      {open && (
        <div className="absolute z-50 mt-1 w-full max-h-56 overflow-auto rounded-md border border-border bg-popover shadow-md">
          {filtered.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              No results
            </div>
          ) : (
            filtered.map((o) => (
              <button
                type="button"
                key={o.value}
                onMouseDown={(e) => {
                  e.preventDefault();
                  onChange(o.value);
                  setOpen(false);
                  setQuery("");
                }}
                className="block w-full text-left px-3 py-2 text-sm hover:bg-muted"
              >
                {o.label}
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Internal settings form — rendered inside the FormProvider so child
 * components (`CacheControlSettings`, etc.) can read/write via
 * `useFormContext`.
 */
function SettingsForm({
  isEditing,
  localModelData,
  guardrailsList,
  tagsList,
  credentialsList,
  modelAccessGroups,
  modelHubData,
  accessToken,
  isWildcardModel,
  isAutoRouter,
  canEditModel,
  onEdit,
  onCancel,
  isSaving,
  showCacheControl,
  setShowCacheControl,
  onEditAutoRouter,
  modelData,
}: {
  isEditing: boolean;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  localModelData: any;
  guardrailsList: string[];
  tagsList: Record<string, Tag>;
  credentialsList: CredentialItem[];
  modelAccessGroups: string[] | null;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  modelHubData: any;
  accessToken: string | null;
  isWildcardModel: boolean;
  isAutoRouter: boolean;
  canEditModel: boolean;
  onEdit: () => void;
  onCancel: () => void;
  isSaving: boolean;
  showCacheControl: boolean;
  setShowCacheControl: (checked: boolean) => void;
  onEditAutoRouter: () => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  modelData: any;
}) {
  const form = useFormContext<ModelSettingsFormValues>();
  const { register, control } = form;

  const tagOptions = Object.values(tagsList).map((tag: Tag) => ({
    value: tag.name,
    label: tag.name,
    title: tag.description || tag.name,
  }));
  const guardrailOptions = guardrailsList.map((name) => ({
    value: name,
    label: name,
  }));
  const accessGroupOptions =
    modelAccessGroups?.map((g) => ({ value: g, label: g })) ?? [];

  return (
    <div className="space-y-4">
      <div className="space-y-4">
        <div>
          <p className="font-medium">Model Name</p>
          {isEditing ? (
            <Input placeholder="Enter model name" {...register("model_name")} />
          ) : (
            <div className="mt-1 p-2 bg-muted rounded">
              {localModelData.model_name}
            </div>
          )}
        </div>

        <div>
          <p className="font-medium">LiteLLM Model Name</p>
          {isEditing ? (
            <Input
              placeholder="Enter LiteLLM model name"
              {...register("litellm_model_name")}
            />
          ) : (
            <div className="mt-1 p-2 bg-muted rounded">
              {localModelData.litellm_model_name}
            </div>
          )}
        </div>

        <div>
          <p className="font-medium">Input Cost (per 1M tokens)</p>
          {isEditing ? (
            <Controller
              control={control}
              name="input_cost"
              render={({ field }) => (
                <NumericalInput
                  placeholder="Enter input cost"
                  value={field.value ?? ""}
                  onChange={(v: unknown) =>
                    field.onChange(
                      typeof v === "number" ? v : v === "" || v == null ? null : Number(v),
                    )
                  }
                />
              )}
            />
          ) : (
            <div className="mt-1 p-2 bg-muted rounded">
              {localModelData?.litellm_params?.input_cost_per_token
                ? (
                    localModelData.litellm_params.input_cost_per_token * 1_000_000
                  ).toFixed(4)
                : localModelData?.model_info?.input_cost_per_token
                  ? (
                      localModelData.model_info.input_cost_per_token * 1_000_000
                    ).toFixed(4)
                  : "Not Set"}
            </div>
          )}
        </div>

        <div>
          <p className="font-medium">Output Cost (per 1M tokens)</p>
          {isEditing ? (
            <Controller
              control={control}
              name="output_cost"
              render={({ field }) => (
                <NumericalInput
                  placeholder="Enter output cost"
                  value={field.value ?? ""}
                  onChange={(v: unknown) =>
                    field.onChange(
                      typeof v === "number" ? v : v === "" || v == null ? null : Number(v),
                    )
                  }
                />
              )}
            />
          ) : (
            <div className="mt-1 p-2 bg-muted rounded">
              {localModelData?.litellm_params?.output_cost_per_token
                ? (
                    localModelData.litellm_params.output_cost_per_token *
                    1_000_000
                  ).toFixed(4)
                : localModelData?.model_info?.output_cost_per_token
                  ? (
                      localModelData.model_info.output_cost_per_token *
                      1_000_000
                    ).toFixed(4)
                  : "Not Set"}
            </div>
          )}
        </div>

        <div>
          <p className="font-medium">API Base</p>
          {isEditing ? (
            <Input placeholder="Enter API base" {...register("api_base")} />
          ) : (
            <div className="mt-1 p-2 bg-muted rounded">
              {localModelData.litellm_params?.api_base || "Not Set"}
            </div>
          )}
        </div>

        <div>
          <p className="font-medium">Custom LLM Provider</p>
          {isEditing ? (
            <Input
              placeholder="Enter custom LLM provider"
              {...register("custom_llm_provider")}
            />
          ) : (
            <div className="mt-1 p-2 bg-muted rounded">
              {localModelData.litellm_params?.custom_llm_provider || "Not Set"}
            </div>
          )}
        </div>

        <div>
          <p className="font-medium">Organization</p>
          {isEditing ? (
            <Input
              placeholder="Enter organization"
              {...register("organization")}
            />
          ) : (
            <div className="mt-1 p-2 bg-muted rounded">
              {localModelData.litellm_params?.organization || "Not Set"}
            </div>
          )}
        </div>

        <div>
          <p className="font-medium">TPM (Tokens per Minute)</p>
          {isEditing ? (
            <Controller
              control={control}
              name="tpm"
              render={({ field }) => (
                <NumericalInput
                  placeholder="Enter TPM"
                  value={field.value ?? ""}
                  onChange={(v: unknown) =>
                    field.onChange(
                      typeof v === "number" ? v : v === "" || v == null ? null : Number(v),
                    )
                  }
                />
              )}
            />
          ) : (
            <div className="mt-1 p-2 bg-muted rounded">
              {localModelData.litellm_params?.tpm || "Not Set"}
            </div>
          )}
        </div>

        <div>
          <p className="font-medium">RPM (Requests per Minute)</p>
          {isEditing ? (
            <Controller
              control={control}
              name="rpm"
              render={({ field }) => (
                <NumericalInput
                  placeholder="Enter RPM"
                  value={field.value ?? ""}
                  onChange={(v: unknown) =>
                    field.onChange(
                      typeof v === "number" ? v : v === "" || v == null ? null : Number(v),
                    )
                  }
                />
              )}
            />
          ) : (
            <div className="mt-1 p-2 bg-muted rounded">
              {localModelData.litellm_params?.rpm || "Not Set"}
            </div>
          )}
        </div>

        <div>
          <p className="font-medium">Max Retries</p>
          {isEditing ? (
            <Controller
              control={control}
              name="max_retries"
              render={({ field }) => (
                <NumericalInput
                  placeholder="Enter max retries"
                  value={field.value ?? ""}
                  onChange={(v: unknown) =>
                    field.onChange(
                      typeof v === "number" ? v : v === "" || v == null ? null : Number(v),
                    )
                  }
                />
              )}
            />
          ) : (
            <div className="mt-1 p-2 bg-muted rounded">
              {localModelData.litellm_params?.max_retries || "Not Set"}
            </div>
          )}
        </div>

        <div>
          <p className="font-medium">Timeout (seconds)</p>
          {isEditing ? (
            <Controller
              control={control}
              name="timeout"
              render={({ field }) => (
                <NumericalInput
                  placeholder="Enter timeout"
                  value={field.value ?? ""}
                  onChange={(v: unknown) =>
                    field.onChange(
                      typeof v === "number" ? v : v === "" || v == null ? null : Number(v),
                    )
                  }
                />
              )}
            />
          ) : (
            <div className="mt-1 p-2 bg-muted rounded">
              {localModelData.litellm_params?.timeout || "Not Set"}
            </div>
          )}
        </div>

        <div>
          <p className="font-medium">Stream Timeout (seconds)</p>
          {isEditing ? (
            <Controller
              control={control}
              name="stream_timeout"
              render={({ field }) => (
                <NumericalInput
                  placeholder="Enter stream timeout"
                  value={field.value ?? ""}
                  onChange={(v: unknown) =>
                    field.onChange(
                      typeof v === "number" ? v : v === "" || v == null ? null : Number(v),
                    )
                  }
                />
              )}
            />
          ) : (
            <div className="mt-1 p-2 bg-muted rounded">
              {localModelData.litellm_params?.stream_timeout || "Not Set"}
            </div>
          )}
        </div>

        <div>
          <p className="font-medium">Model Access Groups</p>
          {isEditing ? (
            <Controller
              control={control}
              name="model_access_group"
              render={({ field }) => (
                <TagsLikeMultiSelect
                  value={(field.value as string[]) || []}
                  onChange={field.onChange}
                  options={accessGroupOptions}
                  placeholder="Select existing groups or type to create new ones"
                />
              )}
            />
          ) : (
            <div className="mt-1 p-2 bg-muted rounded">
              {localModelData.model_info?.access_groups ? (
                Array.isArray(localModelData.model_info.access_groups) ? (
                  localModelData.model_info.access_groups.length > 0 ? (
                    <div className="flex flex-wrap gap-1">
                      {localModelData.model_info.access_groups.map(
                        (group: string, index: number) => (
                          <span
                            key={index}
                            className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                          >
                            {group}
                          </span>
                        ),
                      )}
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
          <p className="font-medium flex items-center">
            Guardrails
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <a
                    href="https://docs.litellm.ai/docs/proxy/guardrails/quick_start"
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <InfoCircleOutlined className="h-3 w-3 ml-1" />
                  </a>
                </TooltipTrigger>
                <TooltipContent>
                  Apply safety guardrails to this model to filter content or enforce policies
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </p>
          {isEditing ? (
            <Controller
              control={control}
              name="guardrails"
              render={({ field }) => (
                <TagsLikeMultiSelect
                  value={(field.value as string[]) || []}
                  onChange={field.onChange}
                  options={guardrailOptions}
                  placeholder="Select existing guardrails or type to create new ones"
                />
              )}
            />
          ) : (
            <div className="mt-1 p-2 bg-muted rounded">
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
          <p className="font-medium flex items-center">
            Attached Knowledge Bases (RAG)
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <a
                    href="https://docs.litellm.ai/docs/completion/knowledgebase"
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <InfoCircleOutlined className="h-3 w-3 ml-1" />
                  </a>
                </TooltipTrigger>
                <TooltipContent>
                  Vector stores used for RAG. Every request to this model will automatically retrieve context from these knowledge bases.
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </p>
          {isEditing ? (
            <Controller
              control={control}
              name="vector_store_ids"
              render={({ field }) => (
                <VectorStoreSelector
                  value={field.value ?? []}
                  onChange={(next) => field.onChange(next)}
                  accessToken={accessToken || ""}
                  placeholder="Select knowledge bases (optional)"
                />
              )}
            />
          ) : (
            <div className="mt-1 p-2 bg-muted rounded">
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
          <p className="font-medium">Tags</p>
          {isEditing ? (
            <Controller
              control={control}
              name="tags"
              render={({ field }) => (
                <TagsLikeMultiSelect
                  value={(field.value as string[]) || []}
                  onChange={field.onChange}
                  options={tagOptions}
                  placeholder="Select existing tags or type to create new ones"
                />
              )}
            />
          ) : (
            <div className="mt-1 p-2 bg-muted rounded">
              {localModelData.litellm_params?.tags ? (
                Array.isArray(localModelData.litellm_params.tags) ? (
                  localModelData.litellm_params.tags.length > 0 ? (
                    <div className="flex flex-wrap gap-1">
                      {localModelData.litellm_params.tags.map(
                        (tag: string, index: number) => (
                          <span
                            key={index}
                            className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800"
                          >
                            {tag}
                          </span>
                        ),
                      )}
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
          <p className="font-medium">Existing Credentials</p>
          {isEditing ? (
            <Controller
              control={control}
              name="litellm_credential_name"
              render={({ field }) => (
                <SearchableSingleSelect
                  value={(field.value as string) || ""}
                  onChange={field.onChange}
                  options={[
                    { value: "", label: "None" },
                    ...credentialsList.map((credential) => ({
                      value: credential.credential_name,
                      label: credential.credential_name,
                    })),
                  ]}
                  placeholder="Select or search for existing credentials"
                />
              )}
            />
          ) : (
            <div className="mt-1 p-2 bg-muted rounded">
              {localModelData.litellm_params?.litellm_credential_name ||
                "Manual"}
            </div>
          )}
        </div>

        {isWildcardModel && (
          <div>
            <p className="font-medium">Health Check Model</p>
            {isEditing ? (
              <Controller
                control={control}
                name="health_check_model"
                render={({ field }) => {
                  const wildcardProvider =
                    modelData.litellm_model_name.split("/")[0];
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  const options = (modelHubData?.data || [])
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    .filter((m: any) =>
                      m.providers?.includes(wildcardProvider) &&
                      m.model_group !== modelData.litellm_model_name,
                    )
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    .map((m: any) => ({
                      value: m.model_group,
                      label: m.model_group,
                    }));
                  return (
                    <SearchableSingleSelect
                      value={(field.value as string) || ""}
                      onChange={field.onChange}
                      options={options}
                      placeholder="Select existing health check model"
                    />
                  );
                }}
              />
            ) : (
              <div className="mt-1 p-2 bg-muted rounded">
                {localModelData.model_info?.health_check_model || "Not Set"}
              </div>
            )}
          </div>
        )}

        {isEditing ? (
          <CacheControlSettings
            showCacheControl={showCacheControl}
            onCacheControlChange={(checked) => setShowCacheControl(checked)}
          />
        ) : (
          <div>
            <p className="font-medium">Cache Control</p>
            <div className="mt-1 p-2 bg-muted rounded">
              {localModelData.litellm_params?.cache_control_injection_points ? (
                <div>
                  <p>Enabled</p>
                  <div className="mt-2">
                    {localModelData.litellm_params.cache_control_injection_points.map(
                      // eslint-disable-next-line @typescript-eslint/no-explicit-any
                      (point: any, i: number) => (
                        <div
                          key={i}
                          className="text-sm text-muted-foreground mb-1"
                        >
                          Location: {point.location},
                          {point.role && <span> Role: {point.role}</span>}
                          {point.index !== undefined && (
                            <span> Index: {point.index}</span>
                          )}
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
          <p className="font-medium">Model Info</p>
          {isEditing ? (
            <Controller
              control={control}
              name="model_info"
              render={({ field }) => (
                <Textarea
                  rows={4}
                  placeholder='{"gpt-4": 100, "claude-v1": 200}'
                  {...field}
                  value={(field.value as string) ?? ""}
                />
              )}
            />
          ) : (
            <div className="mt-1 p-2 bg-muted rounded">
              <pre className="bg-muted p-2 rounded text-xs overflow-auto mt-1">
                {JSON.stringify(localModelData.model_info, null, 2)}
              </pre>
            </div>
          )}
        </div>

        <div>
          <p className="font-medium flex items-center">
            LiteLLM Params
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <a
                    href="https://docs.litellm.ai/docs/completion/input"
                    target="_blank"
                    rel="noopener noreferrer"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <InfoCircleOutlined className="h-3 w-3 ml-1" />
                  </a>
                </TooltipTrigger>
                <TooltipContent>
                  Optional litellm params used for making a litellm.completion() call. Some params are automatically added by LiteLLM.
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </p>
          {isEditing ? (
            <Controller
              control={control}
              name="litellm_extra_params"
              rules={{
                validate: (value: string) => {
                  if (!value) return true;
                  try {
                    JSON.parse(value);
                    return true;
                  } catch {
                    return "Please enter valid JSON";
                  }
                },
              }}
              render={({ field, fieldState }) => (
                <>
                  <Textarea
                    rows={4}
                    placeholder={'{\n  "rpm": 100,\n  "timeout": 0,\n  "stream_timeout": 0\n}'}
                    {...field}
                    value={(field.value as string) ?? ""}
                  />
                  {fieldState.error?.message && (
                    <p className="text-sm text-destructive mt-1">
                      {fieldState.error.message}
                    </p>
                  )}
                </>
              )}
            />
          ) : (
            <div className="mt-1 p-2 bg-muted rounded">
              <pre className="bg-muted p-2 rounded text-xs overflow-auto mt-1">
                {JSON.stringify(localModelData.litellm_params, null, 2)}
              </pre>
            </div>
          )}
        </div>
        <div>
          <p className="font-medium">Team ID</p>
          <div className="mt-1 p-2 bg-muted rounded">
            {modelData.model_info.team_id || "Not Set"}
          </div>
        </div>
      </div>

      {isEditing && (
        <div className="mt-6 flex justify-end gap-2">
          <Button
            variant="secondary"
            type="button"
            onClick={onCancel}
            disabled={isSaving}
          >
            Cancel
          </Button>
          <Button type="submit" disabled={isSaving}>
            {isSaving ? "Saving…" : "Save Changes"}
          </Button>
        </div>
      )}
    </div>
  );
}

export default function ModelInfoView({
  modelId,
  onClose,
  accessToken,
  userID,
  userRole,
  onModelUpdate,
  modelAccessGroups,
}: ModelInfoViewProps) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [localModelData, setLocalModelData] = useState<any>(null);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [isCredentialModalOpen, setIsCredentialModalOpen] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [existingCredential, setExistingCredential] =
    useState<CredentialItem | null>(null);
  const [showCacheControl, setShowCacheControl] = useState(false);
  const [copiedStates, setCopiedStates] = useState<Record<string, boolean>>({});
  const [isAutoRouterModalOpen, setIsAutoRouterModalOpen] = useState(false);
  const [guardrailsList, setGuardrailsList] = useState<string[]>([]);
  const [tagsList, setTagsList] = useState<Record<string, Tag>>({});
  const [credentialsList, setCredentialsList] = useState<CredentialItem[]>([]);

  const { data: rawModelDataResponse, isLoading: isLoadingModel } =
    useModelsInfo(1, 50, undefined, modelId);
  const { data: modelCostMapData } = useModelCostMap();
  const { data: modelHubData } = useModelHub();

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
    const transformed = transformModelData(
      rawModelDataResponse,
      getProviderFromModel,
    );
    return transformed.data[0] || null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rawModelDataResponse, modelCostMapData]);

  const modelData = transformedModelData;

  const canEditModel =
    (userRole === "Admin" || modelData?.model_info?.created_by === userID) &&
    modelData?.model_info?.db_model;
  const isAdmin = userRole === "Admin";
  const isAutoRouter = modelData?.litellm_params?.auto_router_config != null;

  const usingExistingCredential =
    modelData?.litellm_params?.litellm_credential_name != null &&
    modelData?.litellm_params?.litellm_credential_name != undefined;

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

      if (processedModelData?.litellm_params?.cache_control_injection_points) {
        setShowCacheControl(true);
      }
    }
  }, [modelData, localModelData]);

  useEffect(() => {
    const getExistingCredential = async () => {
      if (!accessToken) return;
      if (usingExistingCredential) return;
      let existingCredentialResponse = await credentialGetCall(
        accessToken,
        null,
        modelId,
      );
      setExistingCredential({
        credential_name: existingCredentialResponse["credential_name"],
        credential_values: existingCredentialResponse["credential_values"],
        credential_info: existingCredentialResponse["credential_info"],
      });
    };

    const getModelInfo = async () => {
      if (!accessToken) return;
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

      if (specificModelData?.litellm_params?.cache_control_injection_points) {
        setShowCacheControl(true);
      }
    };

    const fetchGuardrails = async () => {
      if (!accessToken) return;
      try {
        const response = await getGuardrailsList(accessToken);
        const guardrailNames = response.guardrails.map(
          (g: { guardrail_name: string }) => g.guardrail_name,
        );
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [accessToken, modelId]);

  const defaultFormValues = useMemo<ModelSettingsFormValues>(() => {
    if (!localModelData) {
      return {
        model_name: "",
        litellm_model_name: "",
        api_base: "",
        custom_llm_provider: "",
        organization: "",
        tpm: null,
        rpm: null,
        max_retries: null,
        timeout: null,
        stream_timeout: null,
        input_cost: null,
        output_cost: null,
        cache_control: false,
        cache_control_injection_points: [],
        model_access_group: [],
        guardrails: [],
        vector_store_ids: undefined,
        tags: [],
        health_check_model: null,
        litellm_credential_name: "",
        litellm_extra_params: "",
        model_info: "",
      };
    }
    const isWildcard = Boolean(
      localModelData.litellm_model_name?.includes?.("*"),
    );
    return {
      model_name: localModelData.model_name,
      litellm_model_name: localModelData.litellm_model_name,
      api_base: localModelData.litellm_params?.api_base ?? "",
      custom_llm_provider: localModelData.litellm_params?.custom_llm_provider ?? "",
      organization: localModelData.litellm_params?.organization ?? "",
      tpm: localModelData.litellm_params?.tpm ?? null,
      rpm: localModelData.litellm_params?.rpm ?? null,
      max_retries: localModelData.litellm_params?.max_retries ?? null,
      timeout: localModelData.litellm_params?.timeout ?? null,
      stream_timeout: localModelData.litellm_params?.stream_timeout ?? null,
      input_cost: localModelData.litellm_params?.input_cost_per_token
        ? localModelData.litellm_params.input_cost_per_token * 1_000_000
        : localModelData.model_info?.input_cost_per_token * 1_000_000 || null,
      output_cost: localModelData.litellm_params?.output_cost_per_token
        ? localModelData.litellm_params.output_cost_per_token * 1_000_000
        : localModelData.model_info?.output_cost_per_token * 1_000_000 || null,
      cache_control: Boolean(
        localModelData.litellm_params?.cache_control_injection_points,
      ),
      cache_control_injection_points:
        localModelData.litellm_params?.cache_control_injection_points || [],
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
      tags: Array.isArray(localModelData.litellm_params?.tags)
        ? localModelData.litellm_params.tags
        : [],
      health_check_model: isWildcard
        ? localModelData.model_info?.health_check_model ?? null
        : null,
      litellm_credential_name:
        localModelData.litellm_params?.litellm_credential_name || "",
      litellm_extra_params: JSON.stringify(
        Object.fromEntries(
          Object.entries(localModelData.litellm_params || {}).filter(
            ([key]) => key !== "litellm_credential_name",
          ),
        ),
        null,
        2,
      ),
      model_info: modelData
        ? JSON.stringify(modelData.model_info, null, 2)
        : "",
    };
  }, [localModelData, modelData]);

  const form = useForm<ModelSettingsFormValues>({
    defaultValues: defaultFormValues,
    mode: "onSubmit",
  });

  useEffect(() => {
    form.reset(defaultFormValues);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaultFormValues]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleReuseCredential = async (values: any) => {
    if (!accessToken) return;
    const credentialItem = {
      credential_name: values.credential_name,
      model_id: modelId,
      credential_info: {
        custom_llm_provider: localModelData.litellm_params?.custom_llm_provider,
      },
    };
    NotificationsManager.info("Storing credential..");
    await credentialCreateCall(accessToken, credentialItem);
    NotificationsManager.success("Credential stored successfully");
  };

  const handleModelUpdate = form.handleSubmit(async (values) => {
    try {
      if (!accessToken) return;
      setIsSaving(true);

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let parsedExtraParams: Record<string, any> = {};
      try {
        parsedExtraParams = values.litellm_extra_params
          ? JSON.parse(values.litellm_extra_params)
          : {};
        delete parsedExtraParams.litellm_credential_name;
      } catch {
        NotificationsManager.fromBackend("Invalid JSON in LiteLLM Params");
        setIsSaving(false);
        return;
      }

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const updatedLitellmParams: Record<string, any> = {
        ...(values.litellm_params || {}),
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
        input_cost_per_token:
          values.input_cost != null ? values.input_cost / 1_000_000 : undefined,
        output_cost_per_token:
          values.output_cost != null
            ? values.output_cost / 1_000_000
            : undefined,
        tags: values.tags,
      };
      if (values.litellm_credential_name) {
        updatedLitellmParams.litellm_credential_name =
          values.litellm_credential_name;
      } else {
        delete updatedLitellmParams.litellm_credential_name;
      }
      if (values.guardrails) {
        updatedLitellmParams.guardrails = values.guardrails;
      }
      if (values.vector_store_ids && values.vector_store_ids.length > 0) {
        updatedLitellmParams.vector_store_ids = values.vector_store_ids;
      } else if (values.vector_store_ids !== undefined) {
        // User explicitly cleared previously-set vector stores — send [] to clear on backend
        updatedLitellmParams.vector_store_ids = [];
      } else {
        delete updatedLitellmParams.vector_store_ids;
      }

      if (
        values.cache_control &&
        values.cache_control_injection_points?.length > 0
      ) {
        updatedLitellmParams.cache_control_injection_points =
          values.cache_control_injection_points;
      } else {
        delete updatedLitellmParams.cache_control_injection_points;
      }

      let updatedModelInfo;
      try {
        updatedModelInfo = values.model_info
          ? JSON.parse(values.model_info)
          : modelData.model_info;
        if (values.model_access_group) {
          updatedModelInfo = {
            ...updatedModelInfo,
            access_groups: values.model_access_group,
          };
        }
        if (values.health_check_model !== undefined) {
          updatedModelInfo = {
            ...updatedModelInfo,
            health_check_model: values.health_check_model,
          };
        }
      } catch {
        NotificationsManager.fromBackend("Invalid JSON in Model Info");
        setIsSaving(false);
        return;
      }

      const updateData = {
        model_name: values.model_name,
        litellm_params: updatedLitellmParams,
        model_info: updatedModelInfo,
      };

      await modelPatchUpdateCall(accessToken, updateData, modelId);

      const updatedModelData = {
        ...localModelData,
        model_name: values.model_name,
        litellm_model_name: values.litellm_model_name,
        litellm_params: updatedLitellmParams,
        model_info: updatedModelInfo,
      };

      setLocalModelData(updatedModelData);

      if (onModelUpdate) {
        onModelUpdate(updatedModelData);
      }

      NotificationsManager.success("Model settings updated successfully");
      setIsEditing(false);
    } catch (error) {
      console.error("Error updating model:", error);
      NotificationsManager.fromBackend("Failed to update model settings");
    } finally {
      setIsSaving(false);
    }
  });

  if (isLoadingModel) {
    return (
      <div className="p-4">
        <Button variant="ghost" onClick={onClose} className="mb-4">
          <ArrowLeftIcon className="h-4 w-4 mr-2" />
          Back to Models
        </Button>
        <p>Loading...</p>
      </div>
    );
  }

  if (!modelData) {
    return (
      <div className="p-4">
        <Button variant="ghost" onClick={onClose} className="mb-4">
          <ArrowLeftIcon className="h-4 w-4 mr-2" />
          Back to Models
        </Button>
        <p>Model not found</p>
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
          litellm_credential_name:
            localModelData.litellm_params.litellm_credential_name,
          model: localModelData.litellm_model_name,
        },
        {
          mode: localModelData.model_info?.mode,
        },
        localModelData.model_info?.mode,
      );

      if (response.status === "success") {
        NotificationsManager.success("Connection test successful!");
      } else {
        throw new Error(
          response?.result?.error || response?.message || "Unknown error",
        );
      }
    } catch (error) {
      if (error instanceof Error) {
        NotificationsManager.error(
          "Error testing connection: " + truncateString(error.message, 100),
        );
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

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
          <Button
            variant="ghost"
            onClick={onClose}
            className="mb-4"
          >
            <ArrowLeftIcon className="h-4 w-4 mr-2" />
            Back to Models
          </Button>
          <h1 className="text-2xl font-semibold m-0">
            Public Model Name: {getDisplayModelName(modelData)}
          </h1>
          <div className="flex items-center cursor-pointer">
            <p className="text-muted-foreground font-mono m-0">
              {modelData.model_info.id}
            </p>
            <Button
              variant="ghost"
              size="icon"
              className={`h-6 w-6 left-2 z-10 transition-all duration-200 ${
                copiedStates["model-id"]
                  ? "text-green-600 bg-green-50 border-green-200"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              }`}
              onClick={() => copyToClipboard(modelData.model_info.id, "model-id")}
              aria-label="Copy model id"
            >
              {copiedStates["model-id"] ? (
                <CheckIcon size={12} />
              ) : (
                <CopyIcon size={12} />
              )}
            </Button>
          </div>
        </div>
        <div className="flex gap-2">
          <Button
            variant="secondary"
            onClick={handleTestConnection}
            className="flex items-center gap-2"
            data-testid="test-connection-button"
          >
            <RefreshIcon className="h-4 w-4" />
            Test Connection
          </Button>

          <Button
            variant="secondary"
            onClick={() => setIsCredentialModalOpen(true)}
            className="flex items-center gap-2"
            disabled={!isAdmin}
            data-testid="reuse-credentials-button"
          >
            <KeyIcon className="h-4 w-4" />
            Re-use Credentials
          </Button>
          <Button
            variant="secondary"
            onClick={() => setIsDeleteModalOpen(true)}
            className="flex items-center gap-2 text-red-500 border-red-500 hover:text-red-700"
            disabled={!canEditModel}
            data-testid="delete-model-button"
          >
            <TrashIcon className="h-4 w-4" />
            Delete Model
          </Button>
        </div>
      </div>

      <Tabs defaultValue="overview">
        <TabsList className="mb-6">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="raw_json">Raw JSON</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
            <Card className="p-4">
              <p className="text-sm text-muted-foreground">Provider</p>
              <div className="mt-2 flex items-center space-x-2">
                {modelData.provider && (
                  // eslint-disable-next-line @next/next/no-img-element
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
                          "w-4 h-4 rounded-full bg-muted flex items-center justify-center text-xs";
                        fallbackDiv.textContent =
                          modelData.provider?.charAt(0) || "-";
                        parent.replaceChild(fallbackDiv, target);
                      } catch (error) {
                        console.error(
                          "Failed to replace provider logo fallback:",
                          error,
                        );
                      }
                    }}
                  />
                )}
                <h3 className="text-base font-semibold m-0">
                  {modelData.provider || "Not Set"}
                </h3>
              </div>
            </Card>
            <Card className="p-4">
              <p className="text-sm text-muted-foreground">LiteLLM Model</p>
              <div className="mt-2 overflow-hidden">
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div className="break-all text-sm font-medium leading-relaxed cursor-pointer">
                        {modelData.litellm_model_name || "Not Set"}
                      </div>
                    </TooltipTrigger>
                    <TooltipContent>
                      {modelData.litellm_model_name || "Not Set"}
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </div>
            </Card>
            <Card className="p-4">
              <p className="text-sm text-muted-foreground">Pricing</p>
              <div className="mt-2">
                <p className="text-sm">Input: ${modelData.input_cost}/1M tokens</p>
                <p className="text-sm">
                  Output: ${modelData.output_cost}/1M tokens
                </p>
              </div>
            </Card>
          </div>

          <div className="mb-6 text-sm text-muted-foreground flex items-center gap-x-6">
            <div className="flex items-center gap-x-2">
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              Created At{" "}
              {modelData.model_info.created_at
                ? new Date(modelData.model_info.created_at).toLocaleDateString(
                    "en-US",
                    {
                      month: "short",
                      day: "numeric",
                      year: "numeric",
                    },
                  )
                : "Not Set"}
            </div>
            <div className="flex items-center gap-x-2">
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
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

          <Card className="p-4">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold m-0">Model Settings</h2>
              <div className="flex gap-2">
                {isAutoRouter && canEditModel && !isEditing && (
                  <Button
                    onClick={() => setIsAutoRouterModalOpen(true)}
                    className="flex items-center"
                  >
                    Edit Auto Router
                  </Button>
                )}
                {canEditModel ? (
                  !isEditing && (
                    <Button
                      onClick={() => setIsEditing(true)}
                      className="flex items-center"
                    >
                      Edit Settings
                    </Button>
                  )
                ) : (
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span>
                          <InfoCircleOutlined className="h-4 w-4" />
                        </span>
                      </TooltipTrigger>
                      <TooltipContent>
                        Only DB models can be edited. You must be an admin or the creator of the model to edit it.
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                )}
              </div>
            </div>
            {localModelData ? (
              <FormProvider {...form}>
                <form onSubmit={handleModelUpdate}>
                  <SettingsForm
                    isEditing={isEditing}
                    localModelData={localModelData}
                    guardrailsList={guardrailsList}
                    tagsList={tagsList}
                    credentialsList={credentialsList}
                    modelAccessGroups={modelAccessGroups}
                    modelHubData={modelHubData}
                    accessToken={accessToken}
                    isWildcardModel={isWildcardModel}
                    isAutoRouter={isAutoRouter}
                    canEditModel={canEditModel}
                    onEdit={() => setIsEditing(true)}
                    onCancel={() => {
                      form.reset(defaultFormValues);
                      setIsEditing(false);
                    }}
                    isSaving={isSaving}
                    showCacheControl={showCacheControl}
                    setShowCacheControl={setShowCacheControl}
                    onEditAutoRouter={() => setIsAutoRouterModalOpen(true)}
                    modelData={modelData}
                  />
                </form>
              </FormProvider>
            ) : (
              <p>Loading...</p>
            )}
          </Card>
        </TabsContent>

        <TabsContent value="raw_json">
          <Card className="p-4">
            <pre className="bg-muted p-4 rounded text-xs overflow-auto">
              {JSON.stringify(modelData, null, 2)}
            </pre>
          </Card>
        </TabsContent>
      </Tabs>

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
        <Dialog
          open={isCredentialModalOpen}
          onOpenChange={(o) => (!o ? setIsCredentialModalOpen(false) : undefined)}
        >
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Using Existing Credential</DialogTitle>
            </DialogHeader>
            <p>{modelData.litellm_params.litellm_credential_name}</p>
          </DialogContent>
        </Dialog>
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
