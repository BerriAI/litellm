"use client";

import React, { useEffect, useMemo } from "react";
import { Select } from "antd";
import { RobotOutlined } from "@ant-design/icons";
import { TextInput } from "@tremor/react";
import { useRef, useState } from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { useModelsInfo } from "@/app/(dashboard)/hooks/models/useModels";
import { transformModelData } from "@/app/(dashboard)/models-and-endpoints/utils/modelDataTransformer";

export interface DeploymentOption {
  model_id: string;
  model_name: string;
  provider: string;
}

interface DeploymentSelectorProps {
  value?: string | null;
  placeholder?: string;
  onChange?: (deployment: DeploymentOption | null) => void;
  disabled?: boolean;
  style?: React.CSSProperties;
  className?: string;
  showLabel?: boolean;
  labelText?: string;
  excludeIds?: Set<string>;
}

export default function DeploymentSelector({
  value,
  placeholder = "Select a deployment",
  onChange,
  disabled = false,
  style,
  className,
  showLabel = true,
  labelText = "Select Model",
  excludeIds = new Set(),
}: DeploymentSelectorProps) {
  const { accessToken, userId, userRole } = useAuthorized();
  const { data: rawModelData } = useModelsInfo(1, 200);
  const { data: modelCostMapData } = useModelCostMap();
  const [selectedValue, setSelectedValue] = useState<string | undefined>(value ?? undefined);
  const [showCustomModelInput, setShowCustomModelInput] = useState(false);
  const customModelTimeout = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    setSelectedValue(value ?? undefined);
  }, [value]);

  const getProviderFromModel = (model: string) => {
    if (modelCostMapData && typeof modelCostMapData === "object" && model in modelCostMapData) {
      return modelCostMapData[model]?.litellm_provider ?? "openai";
    }
    return "openai";
  };

  const modelData = useMemo(() => {
    if (!rawModelData?.data) return { data: [] };
    return transformModelData(rawModelData, getProviderFromModel);
  }, [rawModelData?.data, modelCostMapData]);

  const deploymentOptions = useMemo(() => {
    if (!modelData?.data) return [];
    return modelData.data
      .filter((m: any) => m.model_info?.id && !excludeIds.has(m.model_info.id))
      .map((m: any) => ({
        model_id: m.model_info.id,
        model_name: m.model_name,
        provider: m.provider || "-",
        label: `${m.provider || "Unknown"} / ${m.model_name}`,
      }));
  }, [modelData?.data, excludeIds]);

  const options = [
    { value: "__custom__", label: "Enter custom model", key: "custom" },
    ...deploymentOptions.map((d) => ({
      value: d.model_id,
      label: d.label,
      key: d.model_id,
    })),
  ];

  const handleChange = (val: string) => {
    if (val === "__custom__") {
      setShowCustomModelInput(true);
      setSelectedValue(undefined);
      onChange?.(null);
    } else {
      setShowCustomModelInput(false);
      setSelectedValue(val);
      const dep = deploymentOptions.find((d) => d.model_id === val);
      if (dep) {
        onChange?.({ model_id: dep.model_id, model_name: dep.model_name, provider: dep.provider });
      }
    }
  };

  const isCustomValue = !!value && !deploymentOptions.some((d) => d.model_id === value);

  const handleCustomModelChange = (inputVal: string) => {
    if (customModelTimeout.current) clearTimeout(customModelTimeout.current);
    customModelTimeout.current = setTimeout(() => {
      if (inputVal.trim()) {
        setSelectedValue(`custom:${inputVal}`);
        onChange?.({
          model_id: inputVal.trim(),
          model_name: inputVal.trim(),
          provider: "custom",
        });
      } else {
        onChange?.(null);
      }
    }, 500);
  };

  return (
    <div>
      {showLabel && (
        <div className="font-medium mb-2 text-gray-700 flex items-center">
          <RobotOutlined className="mr-2" /> {labelText}
        </div>
      )}
      <Select
        value={showCustomModelInput ? undefined : selectedValue}
        placeholder={placeholder}
        onChange={handleChange}
        options={options}
        style={{ width: "100%", ...style }}
        showSearch
        className={`rounded-md ${className || ""}`}
        disabled={disabled}
        filterOption={(input, option) =>
          (option?.label ?? "").toString().toLowerCase().includes(input.toLowerCase())
        }
      />
      {(showCustomModelInput || isCustomValue) && (
        <TextInput
          className="mt-2"
          placeholder="Enter custom model name (e.g. openai/gpt-4)"
          defaultValue={isCustomValue ? value : undefined}
          onValueChange={handleCustomModelChange}
          disabled={disabled}
        />
      )}
    </div>
  );
}
