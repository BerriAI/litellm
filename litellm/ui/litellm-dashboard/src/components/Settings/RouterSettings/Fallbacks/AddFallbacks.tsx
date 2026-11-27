/**
 * Parent component for adding fallbacks to the proxy router config
 * Handles value/onChange logic and form submission
 * Works with forms - reads from and writes to router_settings.fallbacks
 */

import React, { useEffect, useState } from "react";
import { useController } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { toast } from "@/lib/toast";
import { useZodForm } from "@/lib/forms/useZodForm";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import { AddFallbacksModal } from "./AddFallbacksModal";
import { FallbackSelectionForm } from "./FallbackSelectionForm";
import { appendFallbacks, newFallbackGroups, type Fallbacks } from "./mapper";
import { addFallbacksSchema, incompleteFallbackGroups } from "./schema";

export type { FallbackEntry, Fallbacks } from "./mapper";

interface AddFallbacksProps {
  accessToken: string;
  value?: Fallbacks; // Current fallbacks value from form
  onChange?: (fallbacks: Fallbacks) => Promise<void>; // Callback to update form value
  disabledReason?: string; // When set, the button is disabled and this explains why
}

export default function AddFallbacks({ accessToken, value = [], onChange, disabledReason }: AddFallbacksProps) {
  const { t } = useTranslation();
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [modalKey, setModalKey] = useState(0); // Key to force remount of form when modal opens
  const [isSaving, setIsSaving] = useState(false);
  const { control, getValues, reset } = useZodForm(addFallbacksSchema, {
    defaultValues: { groups: newFallbackGroups() },
  });
  const {
    field: { value: groups, onChange: setGroups },
  } = useController({ control, name: "groups" });

  // Reset groups state and increment modal key when modal opens
  useEffect(() => {
    if (isModalVisible) {
      reset({ groups: newFallbackGroups() });
      setModalKey((prev) => prev + 1); // Force remount of form
    }
  }, [isModalVisible, reset]);

  useEffect(() => {
    const loadModels = async () => {
      try {
        const uniqueModels = await fetchAvailableModels(accessToken);
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
    reset({ groups: newFallbackGroups() });
  };

  const handleSaveAll = async () => {
    const values = getValues();
    const result = addFallbacksSchema.safeParse(values);
    if (!result.success) {
      toast.error(
        t("routerSettings.addFallbacks.toasts.incomplete", { count: incompleteFallbackGroups(values.groups).length }),
      );
      return;
    }

    const updatedFallbacks = appendFallbacks(value, result.data);

    // Call onChange to update the form value and wait for it to complete
    if (onChange) {
      setIsSaving(true);
      try {
        await onChange(updatedFallbacks);
        toast.success(t("routerSettings.addFallbacks.toasts.added", { count: result.data.groups.length }));
        handleCancel();
      } catch (error) {
        // Error handling is done in handleFallbacksChange, so we don't need to show another notification here
        console.error("Error saving fallbacks:", error);
      } finally {
        setIsSaving(false);
      }
    } else {
      toast.fromError(t("routerSettings.addFallbacks.toasts.noOnChange"));
    }
  };

  return (
    <div>
      {disabledReason ? (
        <Tooltip>
          <TooltipTrigger render={<span className="inline-flex" tabIndex={0} />}>
            <Button className="mx-auto" disabled aria-disabled="true">
              <span>+</span>
              {t("routerSettings.addFallbacks.addButton")}
            </Button>
          </TooltipTrigger>
          <TooltipContent>{disabledReason}</TooltipContent>
        </Tooltip>
      ) : (
        <Button className="mx-auto" onClick={() => setIsModalVisible(true)}>
          <span>+</span>
          {t("routerSettings.addFallbacks.addButton")}
        </Button>
      )}
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
          <div className="flex items-center justify-end space-x-3 pt-6 mt-6 border-t border-border">
            <Button variant="outline" onClick={handleCancel} disabled={isSaving}>
              {t("routerSettings.addFallbacks.cancel")}
            </Button>
            <Button variant="outline" onClick={handleSaveAll} disabled={groups.length === 0 || isSaving}>
              {isSaving && <UiLoadingSpinner className="size-4" />}
              {isSaving ? t("routerSettings.addFallbacks.saving") : t("routerSettings.addFallbacks.saveAll")}
            </Button>
          </div>
        )}
      </AddFallbacksModal>
    </div>
  );
}
