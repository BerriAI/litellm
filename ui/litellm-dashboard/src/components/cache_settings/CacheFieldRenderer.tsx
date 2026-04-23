"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import React, { useEffect, useMemo, useState } from "react";
import {
  fetchAvailableModels,
  ModelGroup,
} from "../playground/llm_calls/fetch_models";
import NumericalInput from "../shared/numerical_input";

interface CacheFieldRendererProps {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  field: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  currentValue: any;
}

const CacheFieldRenderer: React.FC<CacheFieldRendererProps> = ({
  field,
  currentValue,
}) => {
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>(
    currentValue || "",
  );
  const [modelQuery, setModelQuery] = useState("");
  const { accessToken } = useAuthorized();

  useEffect(() => {
    if (!accessToken) return;
    const loadModels = async () => {
      try {
        const uniqueModels = await fetchAvailableModels(accessToken);
        if (uniqueModels.length > 0) setModelInfo(uniqueModels);
      } catch (error) {
        console.error("Error fetching model info:", error);
      }
    };
    loadModels();
  }, [accessToken]);

  const embeddingModels = useMemo(
    () =>
      modelInfo.filter((option: ModelGroup) => option.mode === "embedding"),
    [modelInfo],
  );
  const filteredEmbeddingModels = useMemo(
    () =>
      embeddingModels.filter((m) =>
        modelQuery
          ? m.model_group.toLowerCase().includes(modelQuery.toLowerCase())
          : true,
      ),
    [embeddingModels, modelQuery],
  );

  if (field.field_type === "Boolean") {
    return (
      <div className="space-y-2">
        <Label className="text-sm font-medium">{field.ui_field_name}</Label>
        <div className="flex items-center">
          <input
            type="checkbox"
            name={field.field_name}
            defaultChecked={currentValue === true || currentValue === "true"}
            className="h-4 w-4 text-primary focus:ring-ring border-input rounded"
          />
          <span className="ml-2 text-sm text-muted-foreground">
            {field.field_description}
          </span>
        </div>
      </div>
    );
  }

  if (field.field_type === "Integer" || field.field_type === "Float") {
    return (
      <div className="space-y-2">
        <Label className="text-sm font-medium">{field.ui_field_name}</Label>
        <NumericalInput
          name={field.field_name}
          type="number"
          step={field.field_type === "Float" ? 0.01 : 1}
          defaultValue={currentValue}
          placeholder={field.field_description}
        />
        <p className="text-xs text-muted-foreground">
          {field.field_description}
        </p>
      </div>
    );
  }

  if (field.field_type === "List") {
    return (
      <div className="space-y-2">
        <Label className="text-sm font-medium">{field.ui_field_name}</Label>
        <Textarea
          name={field.field_name}
          defaultValue={
            typeof currentValue === "object"
              ? JSON.stringify(currentValue, null, 2)
              : currentValue
          }
          placeholder={field.field_description}
          rows={4}
        />
        <p className="text-xs text-muted-foreground">
          {field.field_description}
        </p>
      </div>
    );
  }

  if (field.field_type === "Models_Select") {
    return (
      <div className="space-y-2">
        <Label className="text-sm font-medium">{field.ui_field_name}</Label>
        <Select value={selectedModel} onValueChange={setSelectedModel}>
          <SelectTrigger>
            <SelectValue placeholder="Search and select a model..." />
          </SelectTrigger>
          <SelectContent>
            <div className="px-2 pt-1 pb-2">
              <Input
                placeholder="Search…"
                value={modelQuery}
                onChange={(e) => setModelQuery(e.target.value)}
                className="h-8"
                onKeyDown={(e) => e.stopPropagation()}
              />
            </div>
            {filteredEmbeddingModels.length === 0 ? (
              <div className="py-2 px-3 text-sm text-muted-foreground">
                No matches
              </div>
            ) : (
              filteredEmbeddingModels.map((m) => (
                <SelectItem key={m.model_group} value={m.model_group}>
                  {m.model_group}
                </SelectItem>
              ))
            )}
          </SelectContent>
        </Select>
        {/* Hidden input to capture the value for form submission */}
        <input type="hidden" name={field.field_name} value={selectedModel} />
        {field.field_description && (
          <p className="text-xs text-muted-foreground">
            {field.field_description}
          </p>
        )}
      </div>
    );
  }

  const inputType: "text" | "password" | "email" | "url" =
    field.field_name === "password" || field.field_name.includes("password")
      ? "password"
      : "text";

  return (
    <div className="space-y-2">
      <Label className="text-sm font-medium">{field.ui_field_name}</Label>
      <Input
        name={field.field_name}
        type={inputType}
        defaultValue={currentValue}
        placeholder={field.field_description}
      />
      {field.field_description && (
        <p className="text-xs text-muted-foreground">
          {field.field_description}
        </p>
      )}
    </div>
  );
};

export default CacheFieldRenderer;
