/**
 * Parent component for adding fallbacks to the proxy router config
 * Handles value/onChange logic and form submission
 * Works with forms - reads from and writes to router_settings.fallbacks
 */

import React, { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { toast } from "@/lib/toast";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import { AddFallbacksModal } from "./AddFallbacksModal";
import { FallbackGroup } from "./FallbackGroupConfig";
import { FallbackSelectionForm } from "./FallbackSelectionForm";
import { buildFallbackEntries } from "./fallbackModels";

export type FallbackEntry = { [modelName: string]: string[] };
export type Fallbacks = FallbackEntry[];

interface AddFallbacksProps {
  accessToken: string;
  value?: Fallbacks; // Current fallbacks value from form
  onChange?: (fallbacks: Fallbacks) => Promise<void>; // Callback to update form value
}

export default function AddFallbacks({ accessToken, value = [], onChange }: AddFallbacksProps) {
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [modalKey, setModalKey] = useState(0); // Key to force remount of form when modal opens
  const [isSaving, setIsSaving] = useState(false);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [modelError, setModelError] = useState(false);
  const [groups, setGroups] = useState<FallbackGroup[]>([
    {
      id: "1",
      primaryModel: null,
      fallbackModels: [],
    },
  ]);

  // Reset groups state and increment modal key when modal opens
  useEffect(() => {
    if (isModalVisible) {
      setGroups([
        {
          id: "1",
          primaryModel: null,
          fallbackModels: [],
        },
      ]);
      setModalKey((prev) => prev + 1); // Force remount of form
    }
  }, [isModalVisible]);

  useEffect(() => {
    const loadModels = async () => {
      setIsLoadingModels(true);
      setModelError(false);
      try {
        const uniqueModels = await fetchAvailableModels(accessToken);
        setModelInfo(uniqueModels);
      } catch (error) {
        setModelError(true);
        console.error("Error fetching model info for fallbacks:", error);
      } finally {
        setIsLoadingModels(false);
      }
    };
    if (isModalVisible) {
      loadModels();
    }
  }, [accessToken, isModalVisible]);

  const availableModels = Array.from(new Set(modelInfo.map((option) => option.model_group))).sort();
  const modelsUnavailable = isLoadingModels || modelError;
  const cannotSave = groups.length === 0 || isSaving || modelsUnavailable;

  const handleCancel = () => {
    setIsModalVisible(false);
    // Reset to initial state
    setGroups([
      {
        id: "1",
        primaryModel: null,
        fallbackModels: [],
      },
    ]);
  };

  const handleSaveAll = async () => {
    const result = buildFallbackEntries(groups, modelInfo, value || []);
    if (result.error !== undefined) {
      toast.error(result.error);
      return;
    }

    const updatedFallbacks = [...(value || []), ...result.entries];

    // Call onChange to update the form value and wait for it to complete
    if (onChange) {
      setIsSaving(true);
      try {
        await onChange(updatedFallbacks);
        toast.success(`${result.entries.length} fallback configuration(s) added successfully!`);
        handleCancel();
      } catch (error) {
        // Error handling is done in handleFallbacksChange, so we don't need to show another notification here
        console.error("Error saving fallbacks:", error);
      } finally {
        setIsSaving(false);
      }
    } else {
      toast.fromError("onChange callback not provided");
    }
  };

  return (
    <div>
      <Button className="mx-auto" onClick={() => setIsModalVisible(true)}>
        <span>+</span>
        Add Fallbacks
      </Button>
      <AddFallbacksModal open={isModalVisible} onCancel={handleCancel}>
        {isLoadingModels && <p role="status">Loading models...</p>}
        {modelError && <p role="alert">Could not load models. Close and reopen this dialog to retry.</p>}
        <FallbackSelectionForm
          key={modalKey}
          groups={groups}
          onGroupsChange={setGroups}
          availableModels={availableModels}
          modelInfo={modelInfo}
          maxFallbacks={10}
          maxGroups={5}
        />
        {/* Footer with Cancel and Save buttons */}
        {groups.length > 0 && (
          <div className="flex items-center justify-end space-x-3 pt-6 mt-6 border-t border-border">
            <Button variant="outline" onClick={handleCancel} disabled={isSaving}>
              Cancel
            </Button>
            <Button variant="outline" onClick={handleSaveAll} disabled={cannotSave}>
              {isSaving && <UiLoadingSpinner className="size-4" />}
              {isSaving ? "Saving Configuration..." : "Save All Configurations"}
            </Button>
          </div>
        )}
      </AddFallbacksModal>
    </div>
  );
}
