"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { NumberInput, TextInput } from "@tremor/react";
import { Select } from "antd";
import React, { useEffect, useState } from "react";
import { fetchAvailableModels, ModelGroup } from "../chat_ui/llm_calls/fetch_models";
import NumericalInput from "../shared/numerical_input";

interface CacheFieldRendererProps {
  field: any;
  currentValue: any;
}

const CacheFieldRenderer: React.FC<CacheFieldRendererProps> = ({ field, currentValue }) => {
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>(currentValue || "");
  const { accessToken } = useAuthorized();

  useEffect(() => {
    if (!accessToken) return;

    const loadModels = async () => {
      try {
        const uniqueModels = await fetchAvailableModels(accessToken);
        console.log("Fetched models for selector:", uniqueModels);

        if (uniqueModels.length > 0) {
          setModelInfo(uniqueModels);
        }
      } catch (error) {
        console.error("Error fetching model info:", error);
      }
    };

    loadModels();
  }, [accessToken]);

  if (field.field_type === "Boolean") {
    return (
      <div className="space-y-2">
        <label className="text-sm font-medium text-gray-700">{field.ui_field_name}</label>
        <div className="flex items-center">
          <input
            type="checkbox"
            name={field.field_name}
            defaultChecked={currentValue === true || currentValue === "true"}
            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
          />
          <span className="ml-2 text-sm text-gray-500">{field.field_description}</span>
        </div>
      </div>
    );
  }

  if (field.field_type === "Integer" || field.field_type === "Float") {
    return (
      <div className="space-y-2">
        <label className="text-sm font-medium text-gray-700">{field.ui_field_name}</label>
        <NumericalInput
          name={field.field_name}
          type="number"
          defaultValue={currentValue}
          placeholder={field.field_description}
        />
        <p className="text-xs text-gray-500">{field.field_description}</p>
      </div>
    );
  }

  if (field.field_type === "List") {
    return (
      <div className="space-y-2">
        <label className="text-sm font-medium text-gray-700">{field.ui_field_name}</label>
        <textarea
          name={field.field_name}
          defaultValue={typeof currentValue === "object" ? JSON.stringify(currentValue, null, 2) : currentValue}
          placeholder={field.field_description}
          className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500"
          rows={4}
        />
        <p className="text-xs text-gray-500">{field.field_description}</p>
      </div>
    );
  }

  if (field.field_type === "Models_Select") {
    const embeddingModels = modelInfo
      .filter((option: ModelGroup) => option.mode === "embedding")
      .map((option: ModelGroup) => ({
        value: option.model_group,
        label: option.model_group,
      }));

    return (
      <div className="space-y-2">
        <label className="text-sm font-medium text-gray-700">{field.ui_field_name}</label>
        <Select
          value={selectedModel}
          onChange={setSelectedModel}
          showSearch={true}
          placeholder="Search and select a model..."
          options={embeddingModels}
          style={{ width: "100%" }}
          className="rounded-md"
          filterOption={(input, option) => (option?.label ?? "").toLowerCase().includes(input.toLowerCase())}
        />
        {/* Hidden input to capture the value for form submission */}
        <input type="hidden" name={field.field_name} value={selectedModel} />
        {field.field_description && <p className="text-xs text-gray-500">{field.field_description}</p>}
      </div>
    );
  }

  // Render number input for numeric fields
  if (field.field_type === "Integer" || field.field_type === "Float") {
    return (
      <div className="space-y-2">
        <label className="text-sm font-medium text-gray-700">{field.ui_field_name}</label>
        <NumberInput
          name={field.field_name}
          defaultValue={currentValue}
          placeholder={field.field_description}
          step={field.field_type === "Float" ? 0.01 : 1}
        />
        {field.field_description && <p className="text-xs text-gray-500">{field.field_description}</p>}
      </div>
    );
  }

  // Determine input type for text-based fields
  const inputType: "text" | "password" | "email" | "url" | undefined =
    field.field_name === "password" || field.field_name.includes("password") ? "password" : "text";

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-gray-700">{field.ui_field_name}</label>
      <TextInput
        name={field.field_name}
        type={inputType}
        defaultValue={currentValue}
        placeholder={field.field_description}
      />
      {field.field_description && <p className="text-xs text-gray-500">{field.field_description}</p>}
    </div>
  );
};

export default CacheFieldRenderer;
