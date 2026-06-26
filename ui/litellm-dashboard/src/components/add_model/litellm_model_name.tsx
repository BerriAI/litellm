import React from "react";
import { Form, Select as AntSelect } from "antd";
import { TextInput, Text } from "@tremor/react";
import { Row, Col } from "antd";
import { useTranslation } from "react-i18next";
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
  const { t } = useTranslation();
  const form = Form.useFormInstance();

  const handleModelChange = (value: string | string[]) => {
    // Ensure value is always treated as an array
    const values = Array.isArray(value) ? value : [value];

    // If "all-wildcard" is selected, clear the model_name field
    if (values.includes("all-wildcard")) {
      form.setFieldsValue({ model_name: undefined, model_mappings: [] });
    } else {
      // Get current model value to check if we need to update
      const currentModel = form.getFieldValue("model");

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
        form.setFieldsValue({
          model: values,
          model_mappings: mappings,
        });
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
    form.setFieldsValue({
      model: deploymentName,
      model_mappings: mappings,
    });
  };

  // Handle custom model name changes
  const handleCustomModelNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const customName = e.target.value;

    // Immediately update the model mappings
    const currentMappings = form.getFieldValue("model_mappings") || [];
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

    form.setFieldsValue({ model_mappings: updatedMappings });
  };

  return (
    <>
      <Form.Item
        label={t("addModel.litellmModelName.fieldLabel")}
        tooltip={t("addModel.litellmModelName.fieldTooltip")}
        className="mb-0"
      >
        <Form.Item
          name="model"
          rules={[
            {
              required: true,
              message:
                selectedProvider === Providers.Azure
                  ? t("addModel.litellmModelName.deploymentNameRequired")
                  : t("addModel.litellmModelName.modelRequired"),
            },
          ]}
          noStyle
        >
          {selectedProvider === Providers.Azure ||
          selectedProvider === Providers.OpenAI_Compatible ||
          selectedProvider === Providers.Ollama ? (
            <>
              <TextInput
                placeholder={getPlaceholder(selectedProvider)}
                onChange={selectedProvider === Providers.Azure ? handleAzureDeploymentNameChange : undefined}
              />
            </>
          ) : providerModels.length > 0 ? (
            <AntSelect
              data-testid="model-name-select"
              mode="multiple"
              allowClear
              showSearch
              placeholder={t("addModel.litellmModelName.selectModelsPlaceholder")}
              onChange={handleModelChange}
              optionFilterProp="children"
              filterOption={(input, option) => (option?.label ?? "").toLowerCase().includes(input.toLowerCase())}
              options={[
                {
                  label: t("addModel.litellmModelName.customModelOption"),
                  value: "custom",
                },
                {
                  label: t("addModel.litellmModelName.allModelsWildcard", { provider: selectedProvider }),
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
            <TextInput placeholder={getPlaceholder(selectedProvider)} />
          )}
        </Form.Item>

        <Form.Item noStyle shouldUpdate={(prevValues, currentValues) => prevValues.model !== currentValues.model}>
          {({ getFieldValue }) => {
            const selectedModels = getFieldValue("model") || [];
            const modelArray = Array.isArray(selectedModels) ? selectedModels : [selectedModels];
            return (
              modelArray.includes("custom") && (
                <Form.Item
                  name="custom_model_name"
                  rules={[{ required: true, message: t("addModel.litellmModelName.customModelRequired") }]}
                  className="mt-2"
                >
                  <TextInput
                    placeholder={
                      selectedProvider === Providers.Azure
                        ? t("addModel.litellmModelName.enterAzureDeploymentPlaceholder")
                        : t("addModel.litellmModelName.enterCustomModelPlaceholder")
                    }
                    onChange={handleCustomModelNameChange}
                  />
                </Form.Item>
              )
            );
          }}
        </Form.Item>
      </Form.Item>
      <Row>
        <Col span={10}></Col>
        <Col span={14}>
          <Text className="mb-3 mt-1">
            {selectedProvider === Providers.Azure
              ? t("addModel.litellmModelName.azureDeploymentDescription")
              : t("addModel.litellmModelName.modelDescription")}
          </Text>
        </Col>
      </Row>
    </>
  );
};

export default LiteLLMModelNameField;
