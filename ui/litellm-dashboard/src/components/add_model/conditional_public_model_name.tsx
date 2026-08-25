import React, { useEffect, useMemo } from "react";
import type { ColumnDef } from "@tanstack/react-table";
import { useFormContext, useWatch } from "react-hook-form";
import { DataTable } from "@/components/shared/DataTable";
import { Input } from "@/components/ui/input";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { validatorRules } from "../common_components/formRules";
import { MountedFormField, type MountedFormValues } from "../common_components/MountedFormField";
import { Providers } from "../provider_info_helpers";

interface ModelMapping {
  public_name: string;
  litellm_model: string;
}

const sameMappings = (left: readonly ModelMapping[], right: readonly ModelMapping[]): boolean =>
  left.length === right.length &&
  left.every(
    (mapping, index) =>
      mapping.public_name === right[index].public_name && mapping.litellm_model === right[index].litellm_model,
  );

const modelMappingsRule = {
  validator: async (_: unknown, value: unknown) => {
    if (!value || (value as ModelMapping[]).length === 0) {
      throw new Error("At least one model mapping is required");
    }
    const invalidMappings = (value as ModelMapping[]).filter(
      (mapping) => !mapping.public_name || mapping.public_name.trim() === "",
    );
    if (invalidMappings.length > 0) {
      throw new Error("All model mappings must have valid public names");
    }
  },
};

const tooltipCodeClassName = "rounded-sm bg-background/20 px-1 py-0.5 font-mono text-xs";

const ConditionalPublicModelName: React.FC = () => {
  const form = useFormContext<MountedFormValues>();

  const modelValue = useWatch({ control: form.control, name: "model" }) || [];
  const selectionKey = JSON.stringify(Array.isArray(modelValue) ? modelValue : [modelValue]);
  const selectedModels = useMemo(() => JSON.parse(selectionKey) as string[], [selectionKey]);
  const customModelName = useWatch({ control: form.control, name: "custom_model_name" }) as string | undefined;
  const showPublicModelName = !selectedModels.includes("all-wildcard");
  const selectedProvider = useWatch({ control: form.control, name: "custom_llm_provider" });

  useEffect(() => {
    if (customModelName && selectedModels.includes("custom")) {
      const currentMappings = (form.getValues("model_mappings") as ModelMapping[]) || [];
      const updatedMappings = currentMappings.map((mapping) => {
        if (mapping.public_name === "custom" || mapping.litellm_model === "custom") {
          if (selectedProvider === Providers.Azure) {
            return {
              public_name: customModelName,
              litellm_model: `azure/${customModelName}`,
            };
          }
          return {
            public_name: customModelName,
            litellm_model: customModelName,
          };
        }
        return mapping;
      });
      if (!sameMappings(currentMappings, updatedMappings)) {
        form.setValue("model_mappings", updatedMappings);
      }
    }
  }, [customModelName, selectedModels, selectedProvider, form]);

  // Initial setup of model mappings when models are selected
  useEffect(() => {
    if (selectedModels.length > 0 && !selectedModels.includes("all-wildcard")) {
      // Check if we already have mappings that match the selected models
      const currentMappings = (form.getValues("model_mappings") as ModelMapping[]) || [];

      // Only update if the mappings don't exist or don't match the selected models
      const shouldUpdateMappings =
        currentMappings.length !== selectedModels.length ||
        !selectedModels.every((model) =>
          currentMappings.some((mapping) => {
            if (model === "custom") {
              return mapping.litellm_model === "custom" || mapping.litellm_model === customModelName;
            }
            if (selectedProvider === Providers.Azure) {
              return mapping.litellm_model === `azure/${model}`;
            }
            return mapping.litellm_model === model;
          }),
        );

      if (shouldUpdateMappings) {
        const mappings = selectedModels.map((model: string) => {
          if (model === "custom" && customModelName) {
            if (selectedProvider === Providers.Azure) {
              return {
                public_name: customModelName,
                litellm_model: `azure/${customModelName}`,
              };
            }
            return {
              public_name: customModelName,
              litellm_model: customModelName,
            };
          }
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

        form.setValue("model_mappings", mappings);
      }
    }
  }, [selectedModels, customModelName, selectedProvider, form]);

  if (!showPublicModelName) return null;

  const publicNameTooltipContent = (
    <div className="flex flex-col gap-2 text-left font-normal">
      <div>The name you specify in your API calls to LiteLLM Proxy</div>
      <div>
        <strong>Example:</strong> If you name your public model{" "}
        <code className={tooltipCodeClassName}>example-name</code>, and choose{" "}
        <code className={tooltipCodeClassName}>openai/qwen-plus-latest</code> as the LiteLLM model
      </div>
      <div>
        <strong>Usage:</strong> You make an API call to the LiteLLM proxy with{" "}
        <code className={tooltipCodeClassName}>model = &quot;example-name&quot;</code>
      </div>
      <div>
        <strong>Result:</strong> LiteLLM sends <code className={tooltipCodeClassName}>qwen-plus-latest</code> to the
        provider
      </div>
    </div>
  );

  const liteLLMModelTooltipContent = <div>The model name LiteLLM will send to the LLM API</div>;

  const columns: ColumnDef<ModelMapping>[] = [
    {
      id: "public_name",
      accessorKey: "public_name",
      header: () => (
        <span className="flex items-center">
          Public Model Name
          <SimpleTooltip content={publicNameTooltipContent} width="500px" />
        </span>
      ),
      cell: ({ row }) => {
        return (
          <Input
            value={row.original.public_name}
            onChange={(e) => {
              const newValue = e.target.value;
              const newMappings = [...((form.getValues("model_mappings") as ModelMapping[]) ?? [])];

              // Check conditions for Anthropic -1m suffix handling
              const isAnthropic = selectedProvider === Providers.Anthropic;
              const endsWith1m = newValue.endsWith("-1m");
              const litellmParams = form.getValues("litellm_extra_params") as string | undefined;
              const isLitellmParamsEmpty = !litellmParams || litellmParams.trim() === "";

              let finalPublicName = newValue;

              if (isAnthropic && endsWith1m && isLitellmParamsEmpty) {
                // Set litellm params with extra_headers
                const litellmParamsValue = JSON.stringify(
                  { extra_headers: { "anthropic-beta": "context-1m-2025-08-07" } },
                  null,
                  2,
                );
                form.setValue("litellm_extra_params", litellmParamsValue);

                // Remove -1m suffix from public_name
                finalPublicName = newValue.slice(0, -3); // Remove "-1m" (3 characters)
              }

              newMappings[row.index].public_name = finalPublicName;
              form.setValue("model_mappings", newMappings);
            }}
          />
        );
      },
    },
    {
      id: "litellm_model",
      accessorKey: "litellm_model",
      header: () => (
        <span className="flex items-center">
          LiteLLM Model Name
          <SimpleTooltip content={liteLLMModelTooltipContent} width="360px" />
        </span>
      ),
    },
  ];

  return (
    <MountedFormField
      name="model_mappings"
      label={
        <span className="flex items-center">
          Model Mappings
          <SimpleTooltip content="Map public model names to LiteLLM model names for load balancing" />
        </span>
      }
      required
      rules={{ validate: validatorRules(modelMappingsRule) }}
      className="mb-4"
    >
      {(control) => (
        <DataTable
          data={(control.value as ModelMapping[] | undefined) ?? []}
          columns={columns}
          getRowId={(row) => row.litellm_model}
          size="compact"
        />
      )}
    </MountedFormField>
  );
};

export default ConditionalPublicModelName;
