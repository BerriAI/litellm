import React, { useEffect, useState } from "react";
import { Form, Table } from "antd";
import { TextInput } from "@tremor/react";
import { Tooltip } from "../atoms/index";
import { Providers } from "../provider_info_helpers";

const ConditionalPublicModelName: React.FC = () => {
  const form = Form.useFormInstance();
  const [tableKey, setTableKey] = useState(0); // Add a key to force table re-render

  // Watch the 'model' field for changes and ensure it's always an array
  const modelValue = Form.useWatch("model", form) || [];
  const selectedModels = Array.isArray(modelValue) ? modelValue : [modelValue];
  const customModelName = Form.useWatch("custom_model_name", form);
  const showPublicModelName = !selectedModels.includes("all-wildcard");
  const selectedProvider = Form.useWatch("custom_llm_provider", form);
  // Force table to re-render when custom model name changes
  useEffect(() => {
    if (customModelName && selectedModels.includes("custom")) {
      const currentMappings = form.getFieldValue("model_mappings") || [];
      const updatedMappings = currentMappings.map((mapping: any) => {
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
      form.setFieldValue("model_mappings", updatedMappings);
      setTableKey((prev) => prev + 1); // Force table re-render
    }
  }, [customModelName, selectedModels, selectedProvider, form]);

  // Initial setup of model mappings when models are selected
  useEffect(() => {
    if (selectedModels.length > 0 && !selectedModels.includes("all-wildcard")) {
      // Check if we already have mappings that match the selected models
      const currentMappings = form.getFieldValue("model_mappings") || [];

      // Only update if the mappings don't exist or don't match the selected models
      const shouldUpdateMappings =
        currentMappings.length !== selectedModels.length ||
        !selectedModels.every((model) =>
          currentMappings.some((mapping: { public_name: string; litellm_model: string }) => {
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

        form.setFieldValue("model_mappings", mappings);
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
        <code className="bg-gray-700 px-1 py-0.5 rounded text-xs">example-name</code>, and choose{" "}
        <code className="bg-gray-700 px-1 py-0.5 rounded text-xs">openai/qwen-plus-latest</code> as the LiteLLM model
      </div>
      <div className="mb-2 font-normal">
        <strong>Usage:</strong> You make an API call to the LiteLLM proxy with{" "}
        <code className="bg-gray-700 px-1 py-0.5 rounded text-xs">model = &quot;example-name&quot;</code>
      </div>
      <div className="font-normal">
        <strong>Result:</strong> LiteLLM sends{" "}
        <code className="bg-gray-700 px-1 py-0.5 rounded text-xs">qwen-plus-latest</code> to the provider
      </div>
    </>
  );

  const liteLLMModelTooltipContent = <div>The model name LiteLLM will send to the LLM API</div>;

  const columns = [
    {
      title: (
        <span className="flex items-center">
          Public Model Name
          <Tooltip content={publicNameTooltipContent} width="500px" />
        </span>
      ),
      dataIndex: "public_name",
      key: "public_name",
      render: (text: string, record: any, index: number) => {
        return (
          <TextInput
            value={text}
            onChange={(e) => {
              const newMappings = [...form.getFieldValue("model_mappings")];
              newMappings[index].public_name = e.target.value;
              form.setFieldValue("model_mappings", newMappings);
            }}
          />
        );
      },
    },
    {
      title: (
        <span className="flex items-center">
          LiteLLM Model Name
          <Tooltip content={liteLLMModelTooltipContent} width="360px" />
        </span>
      ),
      dataIndex: "litellm_model",
      key: "litellm_model",
    },
  ];

  return (
    <>
      <Form.Item
        label="Model Mappings"
        name="model_mappings"
        tooltip="Map public model names to LiteLLM model names for load balancing"
        labelCol={{ span: 10 }}
        wrapperCol={{ span: 16 }}
        labelAlign="left"
        rules={[
          {
            required: true,
            validator: async (_, value) => {
              if (!value || value.length === 0) {
                throw new Error("At least one model mapping is required");
              }
              // Check if all mappings have valid public names
              const invalidMappings = value.filter(
                (mapping: any) => !mapping.public_name || mapping.public_name.trim() === "",
              );
              if (invalidMappings.length > 0) {
                throw new Error("All model mappings must have valid public names");
              }
            },
          },
        ]}
      >
        <Table
          key={tableKey} // Add key to force re-render
          dataSource={form.getFieldValue("model_mappings")}
          columns={columns}
          pagination={false}
          size="small"
        />
      </Form.Item>
    </>
  );
};

export default ConditionalPublicModelName;
