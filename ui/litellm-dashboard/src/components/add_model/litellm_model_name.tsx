import React from "react";
import { Select as AntSelect } from "antd";
import { useFormContext, useWatch } from "react-hook-form";
import { Input } from "@/components/ui/input";
import { Row, Col } from "antd";
import { antdRequired } from "../common_components/antdFormRules";
import { labelWithHint } from "@/components/shared/form/LabelWithHint";
import { MountedFormField, type MountedFormValues } from "../common_components/MountedFormField";
import { Providers } from "../provider_info_helpers";

interface LiteLLMModelNameFieldProps {
  selectedProvider: Providers;
  providerModels: string[];
  getPlaceholder: (provider: Providers) => string;
}

const LiteLLMModelNameField: React.FC<LiteLLMModelNameFieldProps> = ({
  selectedProvider,
  providerModels,
  getPlaceholder,
}) => {
  const form = useFormContext<MountedFormValues>();
  const modelValue = useWatch({ control: form.control, name: "model" });
  const selectedModels = Array.isArray(modelValue) ? modelValue : [modelValue];

  const handleModelChange = (value: string | string[]) => {
    // Ensure value is always treated as an array
    const values = Array.isArray(value) ? value : [value];

    // If "all-wildcard" is selected, clear the model_name field
    if (values.includes("all-wildcard")) {
      form.setValue("model_name", undefined);
      form.setValue("model_mappings", []);
    } else {
      // Get current model value to check if we need to update
      const currentModel = form.getValues("model");

      // Only update if the value has actually changed
      if (JSON.stringify(currentModel) !== JSON.stringify(values)) {
        // Create mappings first
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

        // Update both fields in one call to reduce re-renders
        form.setValue("model", values);
        form.setValue("model_mappings", mappings);
      }
    }
  };

  const handleAzureDeploymentNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const deploymentName = e.target.value;

    // Create mapping with Azure-specific format
    const mappings = deploymentName
      ? [
          {
            public_name: deploymentName,
            litellm_model: `azure/${deploymentName}`,
          },
        ]
      : [];

    // Update both fields
    form.setValue("model", deploymentName);
    form.setValue("model_mappings", mappings);
  };

  // Handle custom model name changes
  const handleCustomModelNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const customName = e.target.value;

    // Immediately update the model mappings
    const currentMappings = (form.getValues("model_mappings") as any[]) || [];
    const updatedMappings = currentMappings.map((mapping: any) => {
      if (mapping.public_name === "custom" || mapping.litellm_model === "custom") {
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

    form.setValue("model_mappings", updatedMappings);
  };

  return (
    <>
      <MountedFormField
        name="model"
        label={labelWithHint("LiteLLM Model Name(s)", "The model name LiteLLM will send to the LLM API")}
        required
        rules={{
          validate: {
            required: antdRequired(
              `Please enter ${selectedProvider === Providers.Azure ? "a deployment name" : "at least one model"}.`,
            ),
          },
        }}
        className="mb-0"
      >
        {(control) =>
          selectedProvider === Providers.Azure ||
          selectedProvider === Providers.OpenAI_Compatible ||
          selectedProvider === Providers.Ollama ? (
            <Input
              id={control.id}
              value={(control.value as string | undefined) ?? ""}
              onBlur={control.onBlur}
              placeholder={getPlaceholder(selectedProvider)}
              onChange={(event) => {
                control.onChange(event);
                if (selectedProvider === Providers.Azure) {
                  handleAzureDeploymentNameChange(event);
                }
              }}
            />
          ) : providerModels.length > 0 ? (
            <AntSelect
              id={control.id}
              data-testid="model-name-select"
              mode="multiple"
              allowClear
              showSearch
              placeholder="Select models"
              value={control.value as string[] | undefined}
              onBlur={control.onBlur}
              onChange={(value) => {
                control.onChange(value);
                handleModelChange(value);
              }}
              optionFilterProp="children"
              filterOption={(input, option) => (option?.label ?? "").toLowerCase().includes(input.toLowerCase())}
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
              style={{ width: "100%" }}
            />
          ) : (
            <Input
              id={control.id}
              value={(control.value as string | undefined) ?? ""}
              onChange={control.onChange}
              onBlur={control.onBlur}
              placeholder={getPlaceholder(selectedProvider)}
            />
          )
        }
      </MountedFormField>

      {selectedModels.includes("custom") && (
        <MountedFormField
          name="custom_model_name"
          required
          rules={{ validate: { required: antdRequired("Please enter a custom model name.") } }}
          className="mt-2"
        >
          {(control) => (
            <Input
              id={control.id}
              value={(control.value as string | undefined) ?? ""}
              onBlur={control.onBlur}
              placeholder={
                selectedProvider === Providers.Azure ? "Enter Azure deployment name" : "Enter custom model name"
              }
              onChange={(event) => {
                control.onChange(event);
                handleCustomModelNameChange(event);
              }}
            />
          )}
        </MountedFormField>
      )}
      <Row>
        <Col span={10}></Col>
        <Col span={14}>
          <p className="text-sm mb-3 mt-1">
            {selectedProvider === Providers.Azure
              ? "Your deployment name will be saved as the public model name, and LiteLLM will use 'azure/deployment-name' internally"
              : "The model name LiteLLM will send to the LLM API"}
          </p>
        </Col>
      </Row>
    </>
  );
};

export default LiteLLMModelNameField;
