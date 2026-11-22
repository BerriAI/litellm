import React, { useState, useEffect } from "react";
import { Bot } from "lucide-react";
import { useDebouncedCallback } from "@tanstack/react-pacer/debouncer";
import { useTranslation } from "react-i18next";
import { Input } from "@/components/ui/input";
import { SearchSelect } from "@/components/shared/SearchSelect";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import i18n from "@/i18n/i18n";

const MODEL_SELECT_DEBOUNCE_MS = 500;

interface ModelSelectorProps {
  accessToken: string;
  value?: string | null;
  placeholder?: string;
  onChange?: (value: string | null) => void;
  disabled?: boolean;
  style?: React.CSSProperties;
  className?: string;
  showLabel?: boolean;
  labelText?: string;
}

const ModelSelector: React.FC<ModelSelectorProps> = ({
  accessToken,
  value,
  placeholder = i18n.t("modelSelector.placeholder"),
  onChange,
  disabled = false,
  style,
  className,
  showLabel = true,
  labelText = i18n.t("modelSelector.label"),
}) => {
  const { t } = useTranslation();
  const [selectedModel, setSelectedModel] = useState<string | null>(value ?? null);
  const [showCustomModelInput, setShowCustomModelInput] = useState<boolean>(false);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);

  useEffect(() => {
    setSelectedModel(value ?? null);
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

  const onModelChange = (value: string | null) => {
    if (value === "custom") {
      setShowCustomModelInput(true);
      setSelectedModel(null);
    } else {
      setShowCustomModelInput(false);
      setSelectedModel(value ?? null);
      if (onChange) {
        onChange(value);
      }
    }
  };

  const debouncedSelect = useDebouncedCallback(
    (value: string) => {
      setSelectedModel(value ?? null);
      onChange?.(value);
    },
    { wait: MODEL_SELECT_DEBOUNCE_MS },
  );

  return (
    <div>
      {showLabel && (
        <p className="font-medium block mb-2 text-foreground flex items-center">
          <Bot className="mr-2 size-3.5" /> {labelText}
        </p>
      )}
      <div style={{ width: "100%", ...style }} className={`rounded-md ${className || ""}`}>
        <SearchSelect
          options={[
            ...Array.from(new Set(modelInfo.map((option) => option.model_group))).map((model_group) => ({
              value: model_group,
              label: model_group,
            })),
            { value: "custom", label: t("modelSelector.customOption") },
          ]}
          value={selectedModel}
          placeholder={placeholder}
          onValueChange={onModelChange}
          disabled={disabled}
        />
      </div>
      {showCustomModelInput && (
        <Input
          className="mt-2"
          placeholder={t("modelSelector.customPlaceholder")}
          onChange={(e) => debouncedSelect(e.target.value)}
          disabled={disabled}
        />
      )}
    </div>
  );
};

export default ModelSelector;
