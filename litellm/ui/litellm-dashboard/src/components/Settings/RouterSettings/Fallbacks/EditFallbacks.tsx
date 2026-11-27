/**
 * Modal for editing an existing fallback entry
 * Lets the user add/remove models from a primary model's fallback chain
 * Reuses FallbackGroupConfig with the primary model locked
 */

import { Button } from "@/components/ui/button";
import { useQuery } from "@tanstack/react-query";
import { LoaderCircle, Pencil } from "lucide-react";
import React, { useMemo, useState } from "react";
import { useController } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { fetchAvailableModels } from "@/components/llm_calls/fetch_models";
import { toast } from "@/lib/toast";
import { useZodForm } from "@/lib/forms/useZodForm";
import { AddFallbacksModal } from "./AddFallbacksModal";
import { FallbackGroupConfig } from "./FallbackGroupConfig";
import { replaceFallbacks, toEditFallbackGroup, type FallbackEntry, type Fallbacks } from "./mapper";
import { editFallbacksSchema } from "./schema";

export type { FallbackEntry, Fallbacks } from "./mapper";

interface EditFallbacksProps {
  accessToken: string;
  fallbackEntry: FallbackEntry;
  value: Fallbacks;
  onChange: (fallbacks: Fallbacks) => Promise<void>;
  onClose: () => void;
  maxFallbacks?: number;
}

export default function EditFallbacks({
  accessToken,
  fallbackEntry,
  value,
  onChange,
  onClose,
  maxFallbacks = 10,
}: EditFallbacksProps) {
  const { t } = useTranslation();
  const { control, getValues } = useZodForm(editFallbacksSchema, {
    defaultValues: { group: toEditFallbackGroup(fallbackEntry) },
  });
  const {
    field: { value: group, onChange: setGroup },
  } = useController({ control, name: "group" });
  const [isSaving, setIsSaving] = useState(false);

  const { data: modelGroups = [] } = useQuery({
    queryKey: ["availableModels", "fallbacks"],
    queryFn: () => fetchAvailableModels(accessToken),
    enabled: Boolean(accessToken),
  });

  const availableModels = useMemo(
    () => Array.from(new Set(modelGroups.map((option) => option.model_group))).sort(),
    [modelGroups],
  );

  const handleSave = async () => {
    const result = editFallbacksSchema.safeParse(getValues());
    if (!result.success) {
      return;
    }

    const updatedFallbacks = replaceFallbacks(value, result.data);

    setIsSaving(true);
    try {
      await onChange(updatedFallbacks);
      toast.success(t("routerSettings.editFallbacks.toasts.updated", { model: result.data.group.primaryModel }));
      onClose();
    } catch (error) {
      console.error("Error updating fallbacks:", error);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <AddFallbacksModal open onCancel={onClose}>
      <FallbackGroupConfig
        group={group}
        onChange={setGroup}
        availableModels={availableModels}
        maxFallbacks={maxFallbacks}
        disablePrimaryModel
      />
      <div className="flex items-center justify-end space-x-3 pt-6 mt-6 border-t border-border">
        <Button variant="outline" onClick={onClose} disabled={isSaving}>
          {t("routerSettings.editFallbacks.cancel")}
        </Button>
        <Button onClick={handleSave} disabled={isSaving || group.fallbackModels.length === 0}>
          {isSaving ? <LoaderCircle className="w-4 h-4 animate-spin" /> : <Pencil className="w-4 h-4" />}
          {isSaving ? t("routerSettings.editFallbacks.saving") : t("routerSettings.editFallbacks.save")}
        </Button>
      </div>
    </AddFallbacksModal>
  );
}
