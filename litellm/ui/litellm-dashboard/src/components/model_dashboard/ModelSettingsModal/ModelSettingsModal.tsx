"use client";

import { ConfigType, useProxyConfig } from "@/app/(dashboard)/hooks/proxyConfig/useProxyConfig";
import { StoreModelInDBParams, useStoreModelInDB } from "@/app/(dashboard)/hooks/storeModelInDB/useStoreModelInDB";
import { toast } from "@/lib/toast";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { CircleHelp } from "lucide-react";
import React, { useEffect, useMemo } from "react";
import { useForm } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface ModelSettingsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess?: () => void;
}

const labelWithHint = (label: string, hint: string): React.ReactNode => (
  <>
    {label}
    <Tooltip>
      <TooltipTrigger render={<CircleHelp className="size-3.5 shrink-0 cursor-help text-muted-foreground" />} />
      <TooltipContent>{hint}</TooltipContent>
    </Tooltip>
  </>
);

const ModelSettingsModal: React.FC<ModelSettingsModalProps> = ({ isVisible, onCancel, onSuccess }) => {
  const { t } = useTranslation();
  const { mutateAsync, isPending } = useStoreModelInDB();
  const { data: proxyConfigData, isLoading: isLoadingConfig, refetch } = useProxyConfig(ConfigType.GENERAL_SETTINGS);

  // Refetch config when modal opens to ensure we have the latest values
  useEffect(() => {
    if (isVisible) {
      refetch();
    }
  }, [isVisible, refetch]);

  // Compute initial values from fetched config data
  const initialValues = useMemo<StoreModelInDBParams>(() => {
    if (!proxyConfigData) {
      return {
        store_model_in_db: false,
      };
    }

    const storeModelField = proxyConfigData.find((field) => field.field_name === "store_model_in_db");

    return {
      store_model_in_db: storeModelField?.field_value ?? false,
    };
  }, [proxyConfigData]);

  const form = useForm<StoreModelInDBParams>({ defaultValues: initialValues, values: initialValues });

  const handleFormSubmit = async (formValues: StoreModelInDBParams) => {
    try {
      await mutateAsync(formValues, {
        onSuccess: () => {
          toast.success(t("modelDashboard.modelSettingsModal.toasts.saveSuccess"));
          refetch();
          onSuccess?.();
        },
        onError: (error) => {
          toast.fromError(
            t("modelDashboard.modelSettingsModal.toasts.saveFailed", { error: parseErrorMessage(error) }),
          );
        },
      });
    } catch (error) {
      toast.fromError(t("modelDashboard.modelSettingsModal.toasts.saveFailed", { error: parseErrorMessage(error) }));
    }
  };

  const handleCancel = () => {
    form.reset(initialValues);
    onCancel();
  };

  return (
    <Dialog open={isVisible} onOpenChange={(open) => !open && handleCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-base">{t("modelDashboard.modelSettingsModal.title")}</DialogTitle>
        </DialogHeader>
        <TooltipProvider>
          <form onSubmit={(event) => event.preventDefault()}>
            <FieldGroup>
              <FormField
                control={form.control}
                name="store_model_in_db"
                label={labelWithHint(
                  t("modelDashboard.modelSettingsModal.storeModelInDb.label"),
                  proxyConfigData?.find((f) => f.field_name === "store_model_in_db")?.field_description ||
                    t("modelDashboard.modelSettingsModal.storeModelInDb.hint"),
                )}
              >
                {({ id, value, onChange, onBlur }) =>
                  isLoadingConfig ? (
                    <Skeleton
                      role="status"
                      aria-label={t("modelDashboard.modelSettingsModal.loading")}
                      className="h-[18.4px] w-8 rounded-full"
                    />
                  ) : (
                    <Switch
                      id={id}
                      checked={Boolean(value)}
                      onCheckedChange={onChange}
                      onBlur={onBlur}
                      className="w-fit"
                    />
                  )
                }
              </FormField>
            </FieldGroup>
          </form>
        </TooltipProvider>
        <DialogFooter>
          <Button variant="outline" onClick={handleCancel} disabled={isPending || isLoadingConfig}>
            {t("modelDashboard.modelSettingsModal.cancel")}
          </Button>
          <Button
            disabled={isPending || isLoadingConfig}
            aria-busy={isPending}
            onClick={() => void form.handleSubmit(handleFormSubmit)()}
          >
            {isPending ? t("modelDashboard.modelSettingsModal.saving") : t("modelDashboard.modelSettingsModal.save")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default ModelSettingsModal;
