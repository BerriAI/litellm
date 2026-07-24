/**
 * Modal for editing an existing fallback entry
 * Lets the user add/remove models from a primary model's fallback chain
 * Reuses FallbackGroupConfig with the primary model locked
 */

import { Button } from "antd";
import { useQuery } from "@tanstack/react-query";
import { Pencil } from "lucide-react";
import React, { useMemo, useState } from "react";
import { fetchAvailableModels } from "@/components/llm_calls/fetch_models";
import NotificationManager from "../../../molecules/notifications_manager";
import { AddFallbacksModal } from "./AddFallbacksModal";
import { FallbackGroup, FallbackGroupConfig } from "./FallbackGroupConfig";

export type FallbackEntry = { [modelName: string]: string[] };
export type Fallbacks = FallbackEntry[];

interface EditFallbacksProps {
  accessToken: string;
  fallbackEntry: FallbackEntry;
  value: Fallbacks;
  onChange: (fallbacks: Fallbacks) => Promise<void>;
  onClose: () => void;
  maxFallbacks?: number;
}

const toGroup = (entry: FallbackEntry): FallbackGroup => {
  const primaryModel = Object.keys(entry)[0] ?? null;
  return {
    id: "edit",
    primaryModel,
    fallbackModels: primaryModel ? [...(entry[primaryModel] ?? [])] : [],
  };
};

export default function EditFallbacks({
  accessToken,
  fallbackEntry,
  value,
  onChange,
  onClose,
  maxFallbacks = 10,
}: EditFallbacksProps) {
  const [group, setGroup] = useState<FallbackGroup>(() => toGroup(fallbackEntry));
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
    const primaryModel = group.primaryModel;
    if (!primaryModel) {
      return;
    }

    const updatedFallbacks = (value || []).map((entry) =>
      primaryModel in entry ? { ...entry, [primaryModel]: group.fallbackModels } : entry,
    );

    setIsSaving(true);
    try {
      await onChange(updatedFallbacks);
      NotificationManager.success(`Fallbacks for ${primaryModel} updated successfully!`);
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
      <div className="flex items-center justify-end space-x-3 pt-6 mt-6 border-t border-gray-100">
        <Button type="default" onClick={onClose} disabled={isSaving}>
          Cancel
        </Button>
        <Button
          type="primary"
          icon={<Pencil className="w-4 h-4" />}
          onClick={handleSave}
          disabled={isSaving || group.fallbackModels.length === 0}
          loading={isSaving}
        >
          {isSaving ? "Saving Changes..." : "Save Changes"}
        </Button>
      </div>
    </AddFallbacksModal>
  );
}
