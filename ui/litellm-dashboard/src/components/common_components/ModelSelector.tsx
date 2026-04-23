import React, { useState, useEffect, useRef, useMemo } from "react";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Bot } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  fetchAvailableModels,
  ModelGroup,
} from "../playground/llm_calls/fetch_models";

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
  const [showCustomModelInput, setShowCustomModelInput] =
    useState<boolean>(false);
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
        if (uniqueModels.length > 0) setModelInfo(uniqueModels);
      } catch (error) {
        console.error("Error fetching model info:", error);
      }
    };

    loadModels();
  }, [accessToken]);

  const onModelChange = (v: string) => {
    if (v === "custom") {
      setShowCustomModelInput(true);
      setSelectedModel(undefined);
    } else {
      setShowCustomModelInput(false);
      setSelectedModel(v);
      onChange?.(v);
    }
  };

  const handleCustomModelChange = (v: string) => {
    if (customModelTimeout.current) clearTimeout(customModelTimeout.current);
    customModelTimeout.current = setTimeout(() => {
      setSelectedModel(v);
      onChange?.(v);
    }, 500);
  };

  const uniqueModels = useMemo(
    () => Array.from(new Set(modelInfo.map((o) => o.model_group))),
    [modelInfo],
  );

  return (
    <div>
      {showLabel && (
        <div className="font-medium mb-2 text-foreground flex items-center">
          <Bot className="mr-2 h-4 w-4" /> {labelText}
        </div>
      )}
      <Select
        value={selectedModel}
        onValueChange={onModelChange}
        disabled={disabled}
      >
        <SelectTrigger
          style={{ width: "100%", ...style }}
          className={cn(className)}
        >
          <SelectValue placeholder={placeholder} />
        </SelectTrigger>
        <SelectContent>
          {uniqueModels.map((g) => (
            <SelectItem key={g} value={g}>
              {g}
            </SelectItem>
          ))}
          <SelectItem value="custom">Enter custom model</SelectItem>
        </SelectContent>
      </Select>
      {showCustomModelInput && (
        <Input
          className="mt-2"
          placeholder="Enter custom model name"
          onChange={(e) => handleCustomModelChange(e.target.value)}
          disabled={disabled}
        />
      )}
    </div>
  );
};

export default ModelSelector;
