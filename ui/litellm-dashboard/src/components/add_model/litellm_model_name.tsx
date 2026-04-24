import React from "react";
import { Controller, useFormContext, useWatch } from "react-hook-form";
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
import { X } from "lucide-react";
import { Providers } from "../provider_info_helpers";

interface LiteLLMModelNameFieldProps {
  selectedProvider: Providers;
  providerModels: string[];
  getPlaceholder: (provider: Providers) => string;
}

/**
 * Multi-select rendered with shadcn Select + chip list below. Accepts any
 * `Array<{ label: string; value: string }>` list of options. Mirrors the
 * pattern established in `AccessGroupBaseForm.tsx`.
 */
function MultiSelect({
  value,
  onChange,
  options,
  placeholder,
  testId,
}: {
  value: string[];
  onChange: (next: string[]) => void;
  options: { label: string; value: string }[];
  placeholder: string;
  testId?: string;
}) {
  const selected = value ?? [];
  const remaining = options.filter((o) => !selected.includes(o.value));

  return (
    <div className="space-y-2" data-testid={testId}>
      <Select
        value=""
        onValueChange={(v) => {
          if (v) onChange([...selected, v]);
        }}
      >
        <SelectTrigger>
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {remaining.length === 0 ? (
            <div className="py-2 px-3 text-sm text-muted-foreground">
              No options available
            </div>
          ) : (
            remaining.map((opt) => (
              <SelectItem key={opt.value} value={opt.value}>
                {opt.label}
              </SelectItem>
            ))
          )}
        </SelectContent>
      </Select>
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {selected.map((v) => {
            const opt = options.find((o) => o.value === v);
            return (
              <Badge
                key={v}
                variant="secondary"
                className="flex items-center gap-1"
              >
                {opt?.label ?? v}
                <button
                  type="button"
                  onClick={() =>
                    onChange(selected.filter((s) => s !== v))
                  }
                  className="inline-flex items-center justify-center rounded-full hover:bg-muted-foreground/20"
                  aria-label={`Remove ${opt?.label ?? v}`}
                >
                  <X size={12} />
                </button>
              </Badge>
            );
          })}
        </div>
      )}
    </div>
  );
}

const LiteLLMModelNameField: React.FC<LiteLLMModelNameFieldProps> = ({
  selectedProvider,
  providerModels,
  getPlaceholder,
}) => {
  const { control, setValue, getValues, formState } = useFormContext();
  const selectedModels = (useWatch({ control, name: "model" }) || []) as
    | string[]
    | string;
  const modelArray = Array.isArray(selectedModels)
    ? selectedModels
    : selectedModels
      ? [selectedModels]
      : [];

  const handleModelChange = (values: string[]) => {
    if (values.includes("all-wildcard")) {
      setValue("model_name", undefined);
      setValue("model_mappings", []);
      setValue("model", values);
      return;
    }

    const currentModel = getValues("model");
    if (JSON.stringify(currentModel) !== JSON.stringify(values)) {
      const mappings = values.map((model) => {
        if (selectedProvider === Providers.Azure) {
          return {
            public_name: model,
            litellm_model: `azure/${model}`,
          };
        }
        return {
          public_name: model,
          litellm_model: model,
        };
      });
      setValue("model", values);
      setValue("model_mappings", mappings);
    }
  };

  const handleAzureDeploymentNameChange = (
    e: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const deploymentName = e.target.value;
    const mappings = deploymentName
      ? [
          {
            public_name: deploymentName,
            litellm_model: `azure/${deploymentName}`,
          },
        ]
      : [];
    setValue("model", deploymentName);
    setValue("model_mappings", mappings);
  };

  const handleCustomModelNameChange = (
    e: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const customName = e.target.value;
    setValue("custom_model_name", customName);

    const currentMappings = getValues("model_mappings") || [];
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const updatedMappings = currentMappings.map((mapping: any) => {
      if (
        mapping.public_name === "custom" ||
        mapping.litellm_model === "custom"
      ) {
        if (selectedProvider === Providers.Azure) {
          return {
            public_name: customName,
            litellm_model: `azure/${customName}`,
          };
        }
        return {
          public_name: customName,
          litellm_model: customName,
        };
      }
      return mapping;
    });
    setValue("model_mappings", updatedMappings);
  };

  const showSelectVariant =
    selectedProvider !== Providers.Azure &&
    selectedProvider !== Providers.OpenAI_Compatible &&
    selectedProvider !== Providers.Ollama &&
    providerModels.length > 0;

  const modelError = (formState.errors as Record<string, { message?: string }>)
    .model;
  const customNameError = (
    formState.errors as Record<string, { message?: string }>
  ).custom_model_name;

  const requiredMessage = `Please enter ${
    selectedProvider === Providers.Azure
      ? "a deployment name"
      : "at least one model"
  }.`;

  return (
    <>
      <div className="grid grid-cols-24 gap-2 mb-0 items-start">
        <Label className="col-span-10 pt-2" title="The model name LiteLLM will send to the LLM API">
          LiteLLM Model Name(s)
        </Label>
        <div className="col-span-14 space-y-2">
          {selectedProvider === Providers.Azure ||
          selectedProvider === Providers.OpenAI_Compatible ||
          selectedProvider === Providers.Ollama ? (
            <Controller
              control={control}
              name="model"
              rules={{
                validate: (value) =>
                  (typeof value === "string" && value.length > 0) ||
                  (Array.isArray(value) && value.length > 0) ||
                  requiredMessage,
              }}
              render={({ field }) => (
                <Input
                  placeholder={getPlaceholder(selectedProvider)}
                  value={(field.value as string) ?? ""}
                  onChange={(e) => {
                    field.onChange(e.target.value);
                    if (selectedProvider === Providers.Azure) {
                      handleAzureDeploymentNameChange(e);
                    }
                  }}
                />
              )}
            />
          ) : showSelectVariant ? (
            <Controller
              control={control}
              name="model"
              rules={{
                validate: (value) =>
                  (Array.isArray(value) && value.length > 0) ||
                  requiredMessage,
              }}
              render={({ field }) => (
                <MultiSelect
                  testId="model-name-select"
                  value={(field.value as string[]) ?? []}
                  onChange={(next) => {
                    field.onChange(next);
                    handleModelChange(next);
                  }}
                  placeholder="Select models"
                  options={[
                    {
                      label: "Custom Model Name (Enter below)",
                      value: "custom",
                    },
                    {
                      label: `All ${selectedProvider} Models (Wildcard)`,
                      value: "all-wildcard",
                    },
                    ...providerModels.map((model) => ({
                      label: model,
                      value: model,
                    })),
                  ]}
                />
              )}
            />
          ) : (
            <Controller
              control={control}
              name="model"
              rules={{
                validate: (value) =>
                  (typeof value === "string" && value.length > 0) ||
                  (Array.isArray(value) && value.length > 0) ||
                  requiredMessage,
              }}
              render={({ field }) => (
                <Input
                  placeholder={getPlaceholder(selectedProvider)}
                  value={(field.value as string) ?? ""}
                  onChange={(e) => field.onChange(e.target.value)}
                />
              )}
            />
          )}
          {modelError?.message && (
            <p className="text-sm text-destructive">
              {String(modelError.message)}
            </p>
          )}

          {modelArray.includes("custom") && (
            <div className="mt-2 space-y-1">
              <Controller
                control={control}
                name="custom_model_name"
                rules={{ required: "Please enter a custom model name." }}
                render={({ field }) => (
                  <Input
                    placeholder={
                      selectedProvider === Providers.Azure
                        ? "Enter Azure deployment name"
                        : "Enter custom model name"
                    }
                    value={(field.value as string) ?? ""}
                    onChange={(e) => {
                      field.onChange(e.target.value);
                      handleCustomModelNameChange(e);
                    }}
                  />
                )}
              />
              {customNameError?.message && (
                <p className="text-sm text-destructive">
                  {String(customNameError.message)}
                </p>
              )}
            </div>
          )}
        </div>
      </div>
      <div className="grid grid-cols-24 gap-2">
        <div className="col-span-10" />
        <p className="col-span-14 mb-3 mt-1 text-sm text-muted-foreground">
          {selectedProvider === Providers.Azure
            ? "Your deployment name will be saved as the public model name, and LiteLLM will use 'azure/deployment-name' internally"
            : "The model name LiteLLM will send to the LLM API"}
        </p>
      </div>
    </>
  );
};

export default LiteLLMModelNameField;
