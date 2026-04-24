"use client";

import {
  ConfigType,
  useProxyConfig,
} from "@/app/(dashboard)/hooks/proxyConfig/useProxyConfig";
import {
  StoreModelInDBParams,
  useStoreModelInDB,
} from "@/app/(dashboard)/hooks/storeModelInDB/useStoreModelInDB";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import React, { useEffect, useMemo } from "react";
import { Controller, useForm } from "react-hook-form";

interface ModelSettingsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess?: () => void;
}

type ModelSettingsFormValues = {
  store_model_in_db: boolean;
};

const ModelSettingsModal: React.FC<ModelSettingsModalProps> = ({
  isVisible,
  onCancel,
  onSuccess,
}) => {
  const { mutateAsync, isPending } = useStoreModelInDB();
  const {
    data: proxyConfigData,
    isLoading: isLoadingConfig,
    refetch,
  } = useProxyConfig(ConfigType.GENERAL_SETTINGS);

  useEffect(() => {
    if (isVisible) {
      refetch();
    }
  }, [isVisible, refetch]);

  const initialValues = useMemo<ModelSettingsFormValues>(() => {
    if (!proxyConfigData) {
      return { store_model_in_db: false };
    }

    const storeModelField = proxyConfigData.find(
      (field) => field.field_name === "store_model_in_db",
    );

    return {
      store_model_in_db: Boolean(storeModelField?.field_value ?? false),
    };
  }, [proxyConfigData]);

  const form = useForm<ModelSettingsFormValues>({
    defaultValues: initialValues,
  });

  // Reset form when initial values change (e.g. fresh fetch).
  useEffect(() => {
    form.reset(initialValues);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialValues]);

  const handleFormSubmit = form.handleSubmit(async (formValues) => {
    try {
      await mutateAsync(formValues as StoreModelInDBParams, {
        onSuccess: () => {
          NotificationsManager.success(
            "Model storage settings updated successfully",
          );
          refetch();
          onSuccess?.();
        },
        onError: (error) => {
          NotificationsManager.fromBackend(
            "Failed to save model storage settings: " +
              parseErrorMessage(error),
          );
        },
      });
    } catch (error) {
      NotificationsManager.fromBackend(
        "Failed to save model storage settings: " + parseErrorMessage(error),
      );
    }
  });

  const handleCancel = () => {
    form.reset(initialValues);
    onCancel();
  };

  const storeFieldDescription =
    proxyConfigData?.find((f) => f.field_name === "store_model_in_db")
      ?.field_description ||
    "If enabled, models and config are stored in and loaded from the database.";

  return (
    <Dialog
      open={isVisible}
      onOpenChange={(o) => (!o ? handleCancel() : undefined)}
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="text-base font-semibold">
            Model Settings
          </DialogTitle>
        </DialogHeader>
        <form onSubmit={handleFormSubmit}>
          <div className="flex items-center gap-3 py-2">
            <Label htmlFor="store_model_in_db" title={storeFieldDescription}>
              Store Model in DB
            </Label>
            {isLoadingConfig ? (
              <Skeleton className="h-6 w-full" />
            ) : (
              <Controller
                control={form.control}
                name="store_model_in_db"
                render={({ field }) => (
                  <Switch
                    id="store_model_in_db"
                    checked={Boolean(field.value)}
                    onCheckedChange={(checked) => field.onChange(Boolean(checked))}
                  />
                )}
              />
            )}
          </div>
        </form>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={handleCancel}
            disabled={isPending || isLoadingConfig}
          >
            Cancel
          </Button>
          <Button
            onClick={handleFormSubmit}
            disabled={isPending || isLoadingConfig}
          >
            {isPending ? "Saving..." : "Save Settings"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default ModelSettingsModal;
