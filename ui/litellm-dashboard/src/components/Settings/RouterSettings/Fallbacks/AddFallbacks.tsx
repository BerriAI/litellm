/**
 * Parent component for adding fallbacks to the proxy router config
 * Handles value/onChange logic and form submission
 * Works with forms - reads from and writes to router_settings.fallbacks
 */

import { Button as TremorButton } from "@tremor/react";
import { Button, message } from "antd";
import React, { useEffect, useState } from "react";
import NotificationManager from "../../../molecules/notifications_manager";
import { fetchAvailableModels, ModelGroup } from "../../../playground/llm_calls/fetch_models";
import { AddFallbacksModal } from "./AddFallbacksModal";
import { FallbackGroup } from "./FallbackGroupConfig";
import { FallbackSelectionForm } from "./FallbackSelectionForm";

export type FallbackEntry = { [modelName: string]: string[] };
export type Fallbacks = FallbackEntry[];

interface AddFallbacksProps {
  models?: string[];
  accessToken: string;
  value?: Fallbacks; // Current fallbacks value from form
  onChange?: (fallbacks: Fallbacks) => Promise<void>; // Callback to update form value
}

export default function AddFallbacks({
  models,
  accessToken,
  value = [],
  onChange,
}: AddFallbacksProps) {
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [modalKey, setModalKey] = useState(0); // Key to force remount of form when modal opens
  const [isSaving, setIsSaving] = useState(false);
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
      try {
        const uniqueModels = await fetchAvailableModels(accessToken);
        console.log("Fetched models for fallbacks:", uniqueModels);
        setModelInfo(uniqueModels);
      } catch (error) {
        console.error("Error fetching model info for fallbacks:", error);
      }
    };
    if (isModalVisible) {
      loadModels();
    }
  }, [accessToken, isModalVisible]);

  const availableModels = Array.from(new Set(modelInfo.map((option) => option.model_group))).sort();

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
    // Validation
    const invalidGroups = groups.filter(
      (g) => !g.primaryModel || g.fallbackModels.length === 0,
    );
    if (invalidGroups.length > 0) {
      message.error(
        `Please complete configuration for all groups. ${invalidGroups.length} group(s) incomplete.`,
      );
      return;
    }

    // Create fallback objects in the format expected by the API
    const newFallbacks = groups.map((g) => ({
      [g.primaryModel!]: g.fallbackModels,
    }));

    // Get current fallbacks from form value, or an empty array if it's null/undefined
    const currentFallbacks = value || [];

    // Add new fallbacks to the current fallbacks
    const updatedFallbacks = [...currentFallbacks, ...newFallbacks];

    // Call onChange to update the form value and wait for it to complete
    if (onChange) {
      setIsSaving(true);
      try {
        await onChange(updatedFallbacks);
        NotificationManager.success(`${groups.length} fallback configuration(s) added successfully!`);
        handleCancel();
      } catch (error) {
        // Error handling is done in handleFallbacksChange, so we don't need to show another notification here
        console.error("Error saving fallbacks:", error);
      } finally {
        setIsSaving(false);
      }
    } else {
      NotificationManager.fromBackend("onChange callback not provided");
    }
  };

  return (
    <div>
      <TremorButton
        className="mx-auto"
        onClick={() => setIsModalVisible(true)}
        icon={() => <span className="mr-1">+</span>}
      >
        Add Fallbacks
      </TremorButton>
      <AddFallbacksModal open={isModalVisible} onCancel={handleCancel}>
        <FallbackSelectionForm
          key={modalKey}
          groups={groups}
          onGroupsChange={setGroups}
          availableModels={availableModels}
          maxFallbacks={10}
          maxGroups={5}
        />
        {/* Footer with Cancel and Save buttons */}
        {groups.length > 0 && (
          <div className="flex items-center justify-end space-x-3 pt-6 mt-6 border-t border-gray-100">
            <Button
              type="default"
              onClick={handleCancel}
              disabled={isSaving}
            >
              Cancel
            </Button>
            <Button
              type="default"
              onClick={handleSaveAll}
              disabled={groups.length === 0 || isSaving}
              loading={isSaving}
            >
              {isSaving ? "Saving Configuration..." : "Save All Configurations"}
            </Button>
          </div>
        )}
      </AddFallbacksModal>
    </div>
  );
}
