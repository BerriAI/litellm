import React, { useEffect, useState } from "react";
import { Table } from "antd";
import { useFormContext, useWatch } from "react-hook-form";
import { Input } from "@/components/ui/input";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { antdRules } from "../common_components/antdFormRules";
import { MountedFormField, type MountedFormValues } from "../common_components/MountedFormField";
import { Providers } from "../provider_info_helpers";

interface ModelMapping {
  public_name: string;
  litellm_model: string;
}

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

const ConditionalPublicModelName: React.FC = () => {
  const form = useFormContext<MountedFormValues>();
  const [tableKey, setTableKey] = useState(0); // Add a key to force table re-render

  // Watch the 'model' field for changes and ensure it's always an array
  const modelValue = useWatch({ control: form.control, name: "model" }) || [];
  const selectedModels = Array.isArray(modelValue) ? modelValue : [modelValue];
  const customModelName = useWatch({ control: form.control, name: "custom_model_name" }) as string | undefined;
  const showPublicModelName = !selectedModels.includes("all-wildcard");
  const selectedProvider = useWatch({ control: form.control, name: "custom_llm_provider" });
  // Force table to re-render when custom model name changes
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
      form.setValue("model_mappings", updatedMappings);
      setTableKey((prev) => prev + 1); // Force table re-render
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
        setTableKey((prev) => prev + 1); // Force table re-render
      }
    }
  }, [selectedModels, customModelName, selectedProvider, form]);

  if (!showPublicModelName) return null;

  const publicNameTooltipContent = (
    <>
      <div className="mb-2 font-normal">The name you specify in your API calls to LiteLLM Proxy</div>
      <div className="mb-2 font-normal">
        <strong>Example:</strong> If you name your public model{" "}
        <code className="bg-muted px-1 py-0.5 rounded-sm text-xs">example-name</code>, and choose{" "}
        <code className="bg-muted px-1 py-0.5 rounded-sm text-xs">openai/qwen-plus-latest</code> as the LiteLLM model
      </div>
      <div className="mb-2 font-normal">
        <strong>Usage:</strong> You make an API call to the LiteLLM proxy with{" "}
        <code className="bg-muted px-1 py-0.5 rounded-sm text-xs">model = &quot;example-name&quot;</code>
      </div>
      <div className="font-normal">
        <strong>Result:</strong> LiteLLM sends{" "}
        <code className="bg-muted px-1 py-0.5 rounded-sm text-xs">qwen-plus-latest</code> to the provider
      </div>
    </>
  );

  const liteLLMModelTooltipContent = <div>The model name LiteLLM will send to the LLM API</div>;

  const columns = [
    {
      title: (
        <span className="flex items-center">
          Public Model Name
          <SimpleTooltip content={publicNameTooltipContent} width="500px" />
        </span>
      ),
      dataIndex: "public_name",
      key: "public_name",
      render: (text: string, record: any, index: number) => {
        return (
          <Input
            value={text}
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

              newMappings[index].public_name = finalPublicName;
              form.setValue("model_mappings", newMappings);
            }}
          />
        );
      },
    },
    {
      title: (
        <span className="flex items-center">
          LiteLLM Model Name
          <SimpleTooltip content={liteLLMModelTooltipContent} width="360px" />
        </span>
      ),
      dataIndex: "litellm_model",
      key: "litellm_model",
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
      rules={{ validate: antdRules(modelMappingsRule) }}
      className="mb-4"
    >
      {(control) => (
        <Table
          key={tableKey} // Add key to force re-render
          dataSource={control.value as ModelMapping[] | undefined}
          columns={columns}
          pagination={false}
          size="small"
        />
      )}
    </MountedFormField>
  );
};

export default ConditionalPublicModelName;
