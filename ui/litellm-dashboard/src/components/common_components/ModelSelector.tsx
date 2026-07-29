import React, { useState, useEffect } from "react";
import { TextInput, Text } from "@tremor/react";
import { Select } from "antd";
import { RobotOutlined } from "@ant-design/icons";
import { useDebouncedCallback } from "@tanstack/react-pacer/debouncer";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";

const MODEL_SELECT_DEBOUNCE_MS = 500;

interface ModelSelectorProps {
  accessToken: string;
  value?: string;
  placeholder?: string;
  onChange?: (value: string) => void;
  disabled?: boolean;
  style?: React.CSSProperties;
  className?: string;
  showLabel?: boolean;
  labelText?: string;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  accessToken,
  value,
  placeholder = "Select a Model",
  onChange,
  disabled = false,
  style,
  className,
  showLabel = true,
  labelText = "Select Model",
}) => {
  const [selectedModel, setSelectedModel] = useState<string | undefined>(value);
  const [showCustomModelInput, setShowCustomModelInput] = useState<boolean>(false);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);

  useEffect(() => {
    setSelectedModel(value);
  }, [value]);

  useEffect(() => {
    if (!accessToken) return;

    const loadModels = async () => {
      try {
        const uniqueModels = await fetchAvailableModels(accessToken);

        if (uniqueModels.length > 0) {
          setModelInfo(uniqueModels);
        }
      } catch (error) {
        console.error("Error fetching model info:", error);
      }
    };

    loadModels();
  }, [accessToken]);

  const onModelChange = (value: string) => {
    if (value === "custom") {
      setShowCustomModelInput(true);
      setSelectedModel(undefined);
    } else {
      setShowCustomModelInput(false);
      setSelectedModel(value);
      if (onChange) {
        onChange(value);
      }
    }
  };

  const debouncedSelect = useDebouncedCallback(
    (value: string) => {
      setSelectedModel(value);
      onChange?.(value);
    },
    { wait: MODEL_SELECT_DEBOUNCE_MS },
  );

  return (
    <div>
      {showLabel && (
        <Text className="font-medium block mb-2 text-gray-700 flex items-center">
          <RobotOutlined className="mr-2" /> {labelText}
        </Text>
      )}
      <Select
        value={selectedModel}
        placeholder={placeholder}
        onChange={onModelChange}
        options={[
          ...Array.from(new Set(modelInfo.map((option) => option.model_group))).map((model_group, index) => ({
            value: model_group,
            label: model_group,
            key: index,
          })),
          { value: "custom", label: "Enter custom model", key: "custom" },
        ]}
        style={{ width: "100%", ...style }}
        showSearch={true}
        className={`rounded-md ${className || ""}`}
        disabled={disabled}
      />
      {showCustomModelInput && (
        <TextInput
          className="mt-2"
          placeholder="Enter custom model name"
          onValueChange={debouncedSelect}
          disabled={disabled}
        />
      )}
    </div>
  );
};

export default ModelSelector;
