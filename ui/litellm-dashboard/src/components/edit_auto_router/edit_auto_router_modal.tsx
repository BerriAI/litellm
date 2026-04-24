import React, { useEffect, useState, useCallback, useMemo } from "react";
import { Controller, FormProvider, useForm } from "react-hook-form";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { modelAvailableCall, modelPatchUpdateCall } from "../networking";
import {
  fetchAvailableModels,
  ModelGroup,
} from "../playground/llm_calls/fetch_models";
import RouterConfigBuilder from "../add_model/RouterConfigBuilder";
import NotificationsManager from "../molecules/notifications_manager";

interface EditAutoRouterModalProps {
  isVisible: boolean;
  onCancel: () => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onSuccess: (updatedModel: any) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  modelData: any;
  accessToken: string;
  userRole: string;
}

interface FormValues {
  auto_router_name: string;
  auto_router_default_model: string;
  auto_router_embedding_model: string;
  model_access_group: string[];
}

function TagsInput({
  value,
  onChange,
  options,
  placeholder,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: string[];
  placeholder?: string;
}) {
  const [query, setQuery] = useState("");
  const selected = value ?? [];
  const remaining = useMemo(
    () => options.filter((o) => !selected.includes(o)),
    [options, selected],
  );

  const addTag = (tag: string) => {
    const trimmed = tag.trim();
    if (!trimmed) return;
    if (selected.includes(trimmed)) return;
    onChange([...selected, trimmed]);
    setQuery("");
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={placeholder}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === ",") {
              e.preventDefault();
              addTag(query);
            } else if (e.key === "Backspace" && !query && selected.length) {
              onChange(selected.slice(0, -1));
            }
          }}
        />
      </div>
      {remaining.length > 0 && (
        <Select
          value=""
          onValueChange={(v) => {
            if (v) onChange([...selected, v]);
          }}
        >
          <SelectTrigger>
            <SelectValue placeholder="Pick from existing groups" />
          </SelectTrigger>
          <SelectContent>
            {remaining.map((opt) => (
              <SelectItem key={opt} value={opt}>
                {opt}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      )}
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((v) => (
            <Badge key={v} variant="secondary" className="flex items-center gap-1">
              {v}
              <button
                type="button"
                onClick={() => onChange(selected.filter((s) => s !== v))}
                className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                aria-label={`Remove ${v}`}
              >
                <X size={12} />
              </button>
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

const EditAutoRouterModal: React.FC<EditAutoRouterModalProps> = ({
  isVisible,
  onCancel,
  onSuccess,
  modelData,
  accessToken,
  userRole,
}) => {
  const form = useForm<FormValues>({
    defaultValues: {
      auto_router_name: "",
      auto_router_default_model: "",
      auto_router_embedding_model: "",
      model_access_group: [],
    },
  });
  const [loading, setLoading] = useState(false);
  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [routerConfig, setRouterConfig] = useState<any>(null);

  const initializeForm = useCallback(() => {
    try {
      let parsedConfig = null;
      if (modelData.litellm_params?.auto_router_config) {
        if (typeof modelData.litellm_params.auto_router_config === "string") {
          parsedConfig = JSON.parse(
            modelData.litellm_params.auto_router_config,
          );
        } else {
          parsedConfig = modelData.litellm_params.auto_router_config;
        }
      }

      setRouterConfig(parsedConfig);

      form.reset({
        auto_router_name: modelData.model_name,
        auto_router_default_model:
          modelData.litellm_params?.auto_router_default_model || "",
        auto_router_embedding_model:
          modelData.litellm_params?.auto_router_embedding_model || "",
        model_access_group: modelData.model_info?.access_groups || [],
      });
    } catch (error) {
      console.error("Error parsing auto router config:", error);
      NotificationsManager.fromBackend("Error loading auto router configuration");
    }
  }, [form, modelData]);

  useEffect(() => {
    if (isVisible && modelData) {
      initializeForm();
    }
  }, [isVisible, modelData, initializeForm]);

  useEffect(() => {
    const fetchModelAccessGroups = async () => {
      if (!accessToken) return;
      try {
        const response = await modelAvailableCall(
          accessToken,
          "",
          "",
          false,
          null,
          true,
          true,
        );
        setModelAccessGroups(
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          response["data"].map((model: any) => model["id"]),
        );
      } catch (error) {
        console.error("Error fetching model access groups:", error);
      }
    };

    const loadModels = async () => {
      if (!accessToken) return;
      try {
        const uniqueModels = await fetchAvailableModels(accessToken);
        setModelInfo(uniqueModels);
      } catch (error) {
        console.error("Error fetching model info:", error);
      }
    };

    if (isVisible) {
      fetchModelAccessGroups();
      loadModels();
    }
  }, [isVisible, accessToken]);

  const handleSubmit = form.handleSubmit(async (values) => {
    try {
      setLoading(true);

      const updatedLitellmParams = {
        ...modelData.litellm_params,
        auto_router_config: JSON.stringify(routerConfig),
        auto_router_default_model: values.auto_router_default_model,
        auto_router_embedding_model: values.auto_router_embedding_model || undefined,
      };

      const updatedModelInfo = {
        ...modelData.model_info,
        access_groups: values.model_access_group || [],
      };

      const updateData = {
        model_name: values.auto_router_name,
        litellm_params: updatedLitellmParams,
        model_info: updatedModelInfo,
      };

      await modelPatchUpdateCall(accessToken, updateData, modelData.model_info.id);

      const updatedModelData = {
        ...modelData,
        model_name: values.auto_router_name,
        litellm_params: updatedLitellmParams,
        model_info: updatedModelInfo,
      };

      NotificationsManager.success("Auto router configuration updated successfully");
      onSuccess(updatedModelData);
      onCancel();
    } catch (error) {
      console.error("Error updating auto router:", error);
      NotificationsManager.fromBackend("Failed to update auto router configuration");
    } finally {
      setLoading(false);
    }
  });

  const modelOptions = modelInfo.map((model) => model.model_group);

  return (
    <Dialog
      open={isVisible}
      onOpenChange={(o) => (!o ? onCancel() : undefined)}
    >
      <DialogContent className="max-w-[1000px]">
        <DialogHeader>
          <DialogTitle>Edit Auto Router Configuration</DialogTitle>
        </DialogHeader>
        <div className="space-y-6">
          <p className="text-muted-foreground">
            Edit the auto router configuration including routing logic, default
            models, and access settings.
          </p>

          <FormProvider {...form}>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="auto-router-name">
                  Auto Router Name <span className="text-destructive">*</span>
                </Label>
                <Input
                  id="auto-router-name"
                  placeholder="e.g., auto_router_1, smart_routing"
                  {...form.register("auto_router_name", {
                    required: "Auto router name is required",
                  })}
                />
                {form.formState.errors.auto_router_name && (
                  <p className="text-sm text-destructive">
                    {form.formState.errors.auto_router_name.message as string}
                  </p>
                )}
              </div>

              <div className="w-full">
                <RouterConfigBuilder
                  modelInfo={modelInfo}
                  value={routerConfig}
                  onChange={(config) => {
                    setRouterConfig(config);
                  }}
                />
              </div>

              <div className="space-y-2">
                <Label>
                  Default Model <span className="text-destructive">*</span>
                </Label>
                <Controller
                  control={form.control}
                  name="auto_router_default_model"
                  rules={{ required: "Default model is required" }}
                  render={({ field, fieldState }) => (
                    <>
                      <Select value={field.value || ""} onValueChange={field.onChange}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select a default model" />
                        </SelectTrigger>
                        <SelectContent>
                          {modelOptions.map((m) => (
                            <SelectItem key={m} value={m}>
                              {m}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      {fieldState.error && (
                        <p className="text-sm text-destructive">{fieldState.error.message}</p>
                      )}
                    </>
                  )}
                />
              </div>

              <div className="space-y-2">
                <Label>Embedding Model</Label>
                <Controller
                  control={form.control}
                  name="auto_router_embedding_model"
                  render={({ field }) => (
                    <Select
                      value={field.value || ""}
                      onValueChange={field.onChange}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select an embedding model (optional)" />
                      </SelectTrigger>
                      <SelectContent>
                        {modelOptions.map((m) => (
                          <SelectItem key={m} value={m}>
                            {m}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                />
              </div>

              {userRole === "Admin" && (
                <div className="space-y-2">
                  <Label>Model Access Groups</Label>
                  <p className="text-xs text-muted-foreground">
                    Control who can access this auto router
                  </p>
                  <Controller
                    control={form.control}
                    name="model_access_group"
                    render={({ field }) => (
                      <TagsInput
                        value={field.value ?? []}
                        onChange={field.onChange}
                        options={modelAccessGroups}
                        placeholder="Select existing groups or type to create new ones"
                      />
                    )}
                  />
                </div>
              )}
            </form>
          </FormProvider>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={loading}>
            {loading ? "Saving..." : "Save Changes"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default EditAutoRouterModal;
