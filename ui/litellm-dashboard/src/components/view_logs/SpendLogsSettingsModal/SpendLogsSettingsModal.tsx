"use client";

import {
  ConfigType,
  GeneralSettingsFieldName,
  useDeleteProxyConfigField,
  useProxyConfig,
} from "@/app/(dashboard)/hooks/proxyConfig/useProxyConfig";
import {
  StoreRequestInSpendLogsParams,
  useStoreRequestInSpendLogs,
} from "@/app/(dashboard)/hooks/storeRequestInSpendLogs/useStoreRequestInSpendLogs";
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
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Clock, HelpCircle, LoaderCircle } from "lucide-react";
import React, { useEffect } from "react";
import { useForm, FormProvider, Controller, useFormContext } from "react-hook-form";

interface SpendLogsSettingsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess?: () => void;
}

interface FormValues {
  store_prompts_in_spend_logs: boolean;
  maximum_spend_logs_retention_period?: string;
}

const STORE_PROMPTS_TOOLTIP_FALLBACK =
  "When enabled, prompts will be stored in spend logs for tracking and analysis purposes.";
const RETENTION_PERIOD_TOOLTIP_FALLBACK =
  "Set the maximum retention period for spend logs (e.g., '7d' for 7 days, '30d' for 30 days). Leave empty for no limit.";

const SpendLogsSettingsModal: React.FC<SpendLogsSettingsModalProps> = ({
  isVisible,
  onCancel,
  onSuccess,
}) => {
  const { mutateAsync, isPending } = useStoreRequestInSpendLogs();
  const { mutateAsync: deleteField, isPending: isDeletingField } = useDeleteProxyConfigField();
  const {
    data: proxyConfigData,
    isLoading: isLoadingConfig,
    refetch,
  } = useProxyConfig(ConfigType.GENERAL_SETTINGS);

  const form = useForm<FormValues>({
    defaultValues: {
      store_prompts_in_spend_logs: false,
      maximum_spend_logs_retention_period: "",
    },
  });

  // Reapply backend-provided defaults whenever the modal opens or the config changes.
  useEffect(() => {
    if (!isVisible) return;
    refetch();
  }, [isVisible, refetch]);

  useEffect(() => {
    if (!proxyConfigData) return;
    const storePromptsField = proxyConfigData.find(
      (field) => field.field_name === "store_prompts_in_spend_logs",
    );
    const retentionPeriodField = proxyConfigData.find(
      (field) => field.field_name === "maximum_spend_logs_retention_period",
    );
    form.reset({
      store_prompts_in_spend_logs: Boolean(storePromptsField?.field_value),
      maximum_spend_logs_retention_period:
        (retentionPeriodField?.field_value as string | undefined) ?? "",
    });
  }, [proxyConfigData, form]);

  const storePromptsDescription =
    proxyConfigData?.find((f) => f.field_name === "store_prompts_in_spend_logs")?.field_description ||
    STORE_PROMPTS_TOOLTIP_FALLBACK;
  const retentionPeriodDescription =
    proxyConfigData?.find((f) => f.field_name === "maximum_spend_logs_retention_period")
      ?.field_description || RETENTION_PERIOD_TOOLTIP_FALLBACK;

  const handleFormSubmit = async (formValues: FormValues) => {
    try {
      const retentionPeriodValue = formValues.maximum_spend_logs_retention_period;
      const shouldDeleteRetentionPeriod =
        !retentionPeriodValue ||
        (typeof retentionPeriodValue === "string" && retentionPeriodValue.trim() === "");

      if (shouldDeleteRetentionPeriod) {
        try {
          await deleteField({
            config_type: ConfigType.GENERAL_SETTINGS,
            field_name: GeneralSettingsFieldName.MAXIMUM_SPEND_LOGS_RETENTION_PERIOD,
          });
        } catch (deleteError) {
          // If field doesn't exist, that's okay - continue with update
          console.warn(
            "Failed to delete retention period field (may not exist):",
            deleteError,
          );
        }
      }

      const updateParams: StoreRequestInSpendLogsParams = {
        store_prompts_in_spend_logs: formValues.store_prompts_in_spend_logs,
        ...(retentionPeriodValue &&
          typeof retentionPeriodValue === "string" &&
          retentionPeriodValue.trim() !== "" && {
            maximum_spend_logs_retention_period: retentionPeriodValue,
          }),
      };

      await mutateAsync(updateParams, {
        onSuccess: () => {
          NotificationsManager.success("Spend logs settings updated successfully");
          refetch();
          onSuccess?.();
        },
        onError: (error) => {
          NotificationsManager.fromBackend(
            "Failed to save spend logs settings: " + parseErrorMessage(error),
          );
        },
      });
    } catch (error) {
      NotificationsManager.fromBackend(
        "Failed to save spend logs settings: " + parseErrorMessage(error),
      );
    }
  };

  const handleCancel = () => {
    form.reset();
    onCancel();
  };

  const isSubmitting = isPending || isDeletingField;

  return (
    <Dialog open={isVisible} onOpenChange={(o) => (!o ? handleCancel() : undefined)}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="text-base font-semibold">
            Spend Logs Settings
          </DialogTitle>
        </DialogHeader>
        <FormProvider {...form}>
          <form onSubmit={form.handleSubmit(handleFormSubmit)} className="space-y-4">
            <StorePromptsField
              isLoading={isLoadingConfig}
              description={storePromptsDescription}
            />
            <RetentionPeriodField
              isLoading={isLoadingConfig}
              description={retentionPeriodDescription}
            />
          </form>
        </FormProvider>
        <DialogFooter>
          <Button
            variant="outline"
            onClick={handleCancel}
            disabled={isSubmitting || isLoadingConfig}
          >
            Cancel
          </Button>
          <Button
            onClick={form.handleSubmit(handleFormSubmit)}
            disabled={isSubmitting || isLoadingConfig}
            data-loading={isSubmitting || undefined}
          >
            {isSubmitting ? (
              <>
                <LoaderCircle className="mr-2 h-4 w-4 animate-spin" />
                Saving...
              </>
            ) : (
              "Save Settings"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

function FieldTooltip({ text }: { text: string }) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            className="text-muted-foreground hover:text-foreground"
            aria-label="More info"
          >
            <HelpCircle className="h-3.5 w-3.5" />
          </button>
        </TooltipTrigger>
        <TooltipContent className="max-w-xs text-xs">{text}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

function StorePromptsField({
  isLoading,
  description,
}: {
  isLoading: boolean;
  description: string;
}) {
  const { control } = useFormContext<FormValues>();
  return (
    <div className="flex items-start justify-between gap-4">
      <div className="flex items-center gap-1">
        <Label htmlFor="store-prompts-switch" className="text-sm font-medium">
          Store Prompts in Spend Logs
        </Label>
        <FieldTooltip text={description} />
      </div>
      <div>
        {isLoading ? (
          <Skeleton className="h-6 w-12" />
        ) : (
          <Controller
            control={control}
            name="store_prompts_in_spend_logs"
            render={({ field }) => (
              <Switch
                id="store-prompts-switch"
                checked={field.value}
                onCheckedChange={field.onChange}
              />
            )}
          />
        )}
      </div>
    </div>
  );
}

function RetentionPeriodField({
  isLoading,
  description,
}: {
  isLoading: boolean;
  description: string;
}) {
  const { register } = useFormContext<FormValues>();
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1">
        <Label htmlFor="retention-period-input" className="text-sm font-medium">
          Maximum Spend Logs Retention Period (Optional)
        </Label>
        <FieldTooltip text={description} />
      </div>
      {isLoading ? (
        <Skeleton className="h-9 w-full" />
      ) : (
        <div className="relative">
          <Clock className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
          <Input
            id="retention-period-input"
            placeholder="e.g., 7d, 30d"
            className="pl-8"
            {...register("maximum_spend_logs_retention_period")}
          />
        </div>
      )}
    </div>
  );
}

export default SpendLogsSettingsModal;
