import React, { useState, useEffect, useRef } from "react";
import { TextInput, Text } from "@tremor/react";
import { Select } from "antd";
import { RobotOutlined } from "@ant-design/icons";
import { useTranslation } from "react-i18next";
import { fetchAvailableModels, ModelGroup } from "../playground/llm_calls/fetch_models";

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
  placeholder,
  onChange,
  disabled = false,
  style,
  className,
  showLabel = true,
  labelText,
}) => {
  const { t } = useTranslation();
  const [selectedModel, setSelectedModel] = useState<string | undefined>(value);
  const [showCustomModelInput, setShowCustomModelInput] = useState<boolean>(false);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const customModelTimeout = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    setSelectedModel(value);
  }, [value]);

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

  const handleCustomModelChange = (value: string) => {
    // Using setTimeout to create a simple debounce effect
    if (customModelTimeout.current) {
      clearTimeout(customModelTimeout.current);
    }

    customModelTimeout.current = setTimeout(() => {
      setSelectedModel(value);
      if (onChange) {
        onChange(value);
      }
    }, 500); // 500ms delay after typing stops
  };

  return (
    <div>
      {showLabel && (
        <Text className="font-medium block mb-2 text-gray-700 flex items-center">
          <RobotOutlined className="mr-2" /> {labelText ?? t("commonComponents.modelSelector.labelText")}
        </Text>
      )}
      <Select
        value={selectedModel}
        placeholder={placeholder ?? t("commonComponents.modelSelector.placeholder")}
        onChange={onModelChange}
        options={[
          ...Array.from(new Set(modelInfo.map((option) => option.model_group))).map((model_group, index) => ({
            value: model_group,
            label: model_group,
            key: index,
          })),
          { value: "custom", label: t("commonComponents.modelSelector.enterCustomModel"), key: "custom" },
        ]}
        style={{ width: "100%", ...style }}
        showSearch={true}
        className={`rounded-md ${className || ""}`}
        disabled={disabled}
      />
      {showCustomModelInput && (
        <TextInput
          className="mt-2"
          placeholder={t("commonComponents.modelSelector.customModelPlaceholder")}
          onValueChange={handleCustomModelChange}
          disabled={disabled}
        />
      )}
    </div>
  );
};

export default ModelSelector;
