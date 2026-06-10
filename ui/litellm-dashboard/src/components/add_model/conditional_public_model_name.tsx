import React, { useEffect, useState } from "react";
import { Form, Table } from "antd";
import { TextInput } from "@tremor/react";
import { Trans, useTranslation } from "react-i18next";
import { Tooltip } from "../atoms/index";
import { Providers } from "../provider_info_helpers";

const ConditionalPublicModelName: React.FC = () => {
  const { t } = useTranslation();
  const form = Form.useFormInstance();
  const [tableKey, setTableKey] = useState(0);

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
      <div className="mb-2 font-normal">{t("addModel.conditionalPublicModelName.publicModelNameTooltip")}</div>
      <div className="mb-2 font-normal">
        <Trans
          i18nKey="addModel.conditionalPublicModelName.tooltipExample"
          components={{
            strong: <strong />,
            publicName: <code className="bg-gray-700 px-1 py-0.5 rounded text-xs" />,
            modelName: <code className="bg-gray-700 px-1 py-0.5 rounded text-xs" />,
          }}
        />
      </div>
      <div className="mb-2 font-normal">
        <strong>{t("addModel.conditionalPublicModelName.tooltipUsage")}</strong>{" "}
        <code className="bg-gray-700 px-1 py-0.5 rounded text-xs">
          {t("addModel.conditionalPublicModelName.tooltipUsageCode")}
        </code>
      </div>
      <div className="font-normal">
        <strong>{t("addModel.conditionalPublicModelName.tooltipResult")}</strong>{" "}
        <code className="bg-gray-700 px-1 py-0.5 rounded text-xs">
          {t("addModel.conditionalPublicModelName.tooltipResultCode")}
        </code>{" "}
        {t("addModel.conditionalPublicModelName.tooltipResultSuffix")}
      </div>
    </>
  );

  const liteLLMModelTooltipContent = <div>{t("addModel.conditionalPublicModelName.litellmModelNameTooltip")}</div>;

  const columns = [
    {
      title: (
        <span className="flex items-center">
          {t("addModel.conditionalPublicModelName.publicModelNameColumn")}
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
              const newValue = e.target.value;
              const newMappings = [...form.getFieldValue("model_mappings")];

              // Check conditions for Anthropic -1m suffix handling
              const isAnthropic = selectedProvider === Providers.Anthropic;
              const endsWith1m = newValue.endsWith("-1m");
              const litellmParams = form.getFieldValue("litellm_extra_params");
              const isLitellmParamsEmpty = !litellmParams || litellmParams.trim() === "";

              let finalPublicName = newValue;

              if (isAnthropic && endsWith1m && isLitellmParamsEmpty) {
                // Set litellm params with extra_headers
                const litellmParamsValue = JSON.stringify(
                  { extra_headers: { "anthropic-beta": "context-1m-2025-08-07" } },
                  null,
                  2,
                );
                form.setFieldValue("litellm_extra_params", litellmParamsValue);

                // Remove -1m suffix from public_name
                finalPublicName = newValue.slice(0, -3); // Remove "-1m" (3 characters)
              }

              newMappings[index].public_name = finalPublicName;
              form.setFieldValue("model_mappings", newMappings);
            }}
          />
        );
      },
    },
    {
      title: (
        <span className="flex items-center">
          {t("addModel.conditionalPublicModelName.litellmModelNameColumn")}
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
        label={t("addModel.conditionalPublicModelName.modelMappingsLabel")}
        name="model_mappings"
        tooltip={t("addModel.conditionalPublicModelName.modelMappingsTooltip")}
        labelCol={{ span: 10 }}
        wrapperCol={{ span: 16 }}
        labelAlign="left"
        rules={[
          {
            required: true,
            validator: async (_, value) => {
              if (!value || value.length === 0) {
                throw new Error(t("addModel.conditionalPublicModelName.mappingRequired"));
              }
              const invalidMappings = value.filter(
                (mapping: any) => !mapping.public_name || mapping.public_name.trim() === "",
              );
              if (invalidMappings.length > 0) {
                throw new Error(t("addModel.conditionalPublicModelName.publicNameRequired"));
              }
            },
          },
        ]}
      >
        <Table
          key={tableKey}
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
