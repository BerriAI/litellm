import { useLanguage } from "@/contexts/LanguageContext";
import React, { useState, useEffect } from "react";
import { CircleCheck, FileDown } from "lucide-react";
import { z } from "zod";
import { getGlobalLitellmHeaderName } from "@/components/networking";
import { toast } from "@/lib/toast";
import { ApiError, deriveErrorMessage, unwrapProxyErrorMessage } from "@/lib/http/client";
import TranslatedMessage from "@/components/shared/TranslatedMessage";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { PasswordInput } from "@/components/shared/PasswordInput";
import { FieldGroup } from "@/components/ui/field";
import { FormField } from "@/components/shared/form/FormField";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

// 固定原文校验仅在字段显示时翻译，不因语言切换重新校验或清空凭据。
const cloudZeroSettingsSchema = z.object({
  api_key: z.string().min(1, "Please enter your CloudZero API key"),
  connection_id: z.string().min(1, "Please enter the CloudZero connection ID"),
});

// 复用已知错误包装的解包，不把无法识别的完整响应或调试字段展示给用户。
const cloudZeroErrorMessage = (data: unknown): string | undefined => {
  const message = deriveErrorMessage(data);
  return typeof message === "string" && message !== JSON.stringify(data)
    ? unwrapProxyErrorMessage(message) || undefined
    : undefined;
};

interface CloudZeroExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  accessToken: string | null;
}

type CloudZeroSettings = z.output<typeof cloudZeroSettingsSchema>;

interface CloudZeroSettingsView {
  api_key_masked: string;
  connection_id: string;
  status: string;
}

type ExportType = "cloudzero" | "csv";

const CloudZeroExportModal: React.FC<CloudZeroExportModalProps> = ({ isOpen, onClose, accessToken }) => {
  const { t } = useLanguage();
  const form = useZodForm(cloudZeroSettingsSchema, {
    defaultValues: { api_key: "", connection_id: "" },
  });
  const [loading, setLoading] = useState(false);
  const [existingSettings, setExistingSettings] = useState<CloudZeroSettingsView | null>(null);
  const [settingsLoading, setSettingsLoading] = useState(false);
  const [exportType, setExportType] = useState<ExportType>("cloudzero");
  const [exportLoading, setExportLoading] = useState(false);

  // Load existing settings when modal opens
  useEffect(() => {
    if (isOpen && accessToken) {
      loadExistingSettings();
    }
  }, [isOpen, accessToken]);

  const loadExistingSettings = async () => {
    setSettingsLoading(true);
    try {
      const response = await fetch("/cloudzero/settings", {
        method: "GET",
        headers: {
          [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
      });

      if (response.ok) {
        const settings = await response.json();
        setExistingSettings(settings);
        // Pre-populate form with existing settings (except masked API key)
        form.setValue("connection_id", settings.connection_id);
      } else if (response.status !== 404) {
        // 404 means no settings configured yet, which is fine
        const errorData = await response.json().catch(() => undefined);
        const detail = cloudZeroErrorMessage(errorData);
        toast.fromError(new ApiError(detail ?? "Failed to load existing settings", response.status, errorData), {
          description: detail ? (
            <TranslatedMessage message="Failed to load existing settings: {error}" values={{ error: detail }} />
          ) : (
            <TranslatedMessage message="Failed to load existing settings" />
          ),
        });
      }
    } catch (error) {
      console.error("Error loading CloudZero settings:", error);
      toast.fromError(error, { description: <TranslatedMessage message="Failed to load existing settings" /> });
    } finally {
      setSettingsLoading(false);
    }
  };

  const handleSaveCloudZeroSettings = async (values: CloudZeroSettings) => {
    if (!accessToken) {
      toast.fromError("No access token available", {
        description: <TranslatedMessage message="No access token available" />,
      });
      return;
    }

    setLoading(true);
    try {
      const endpoint = existingSettings ? "/cloudzero/settings" : "/cloudzero/init";
      const method = existingSettings ? "PUT" : "POST";

      // Add default timezone for backend compatibility
      const payload = {
        ...values,
        timezone: "UTC",
      };

      const response = await fetch(endpoint, {
        method,
        headers: {
          [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      // 非 JSON 错误仍保留 HTTP 状态，用本地提示替代解析细节。
      const data = response.ok ? await response.json() : await response.json().catch(() => undefined);

      if (response.ok) {
        toast.success(data.message || <TranslatedMessage message="CloudZero settings saved successfully" />);
        setExistingSettings({
          api_key_masked: values.api_key.substring(0, 4) + "****" + values.api_key.slice(-4),
          connection_id: values.connection_id,
          status: "configured",
        });
        return true;
      } else {
        const detail = cloudZeroErrorMessage(data);
        toast.fromError(new ApiError(detail ?? "Failed to save CloudZero settings", response.status, data), {
          description: detail ?? <TranslatedMessage message="Failed to save CloudZero settings" />,
        });
        return false;
      }
    } catch (error) {
      console.error("Error saving CloudZero settings:", error);
      toast.fromError(error, { description: <TranslatedMessage message="Failed to save CloudZero settings" /> });
      return false;
    } finally {
      setLoading(false);
    }
  };

  const handleExportCloudZero = async () => {
    if (!accessToken) {
      toast.fromError("No access token available", {
        description: <TranslatedMessage message="No access token available" />,
      });
      return;
    }

    setExportLoading(true);
    try {
      const response = await fetch("/cloudzero/export", {
        method: "POST",
        headers: {
          [getGlobalLitellmHeaderName()]: `Bearer ${accessToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          limit: 100000,
          operation: "replace_hourly",
        }),
      });

      const data = response.ok ? await response.json() : await response.json().catch(() => undefined);

      if (response.ok) {
        toast.success(data.message || <TranslatedMessage message="Export to CloudZero completed successfully" />);
        onClose();
      } else {
        const detail = cloudZeroErrorMessage(data);
        toast.fromError(new ApiError(detail ?? "Failed to export to CloudZero", response.status, data), {
          description: detail ?? <TranslatedMessage message="Failed to export to CloudZero" />,
        });
      }
    } catch (error) {
      console.error("Error exporting to CloudZero:", error);
      toast.fromError(error, { description: <TranslatedMessage message="Failed to export to CloudZero" /> });
    } finally {
      setExportLoading(false);
    }
  };

  const handleExportCSV = async () => {
    setExportLoading(true);
    try {
      // TODO: Implement CSV export functionality
      toast.info(<TranslatedMessage message="CSV export functionality coming soon!" />);
      onClose();
    } catch (error) {
      console.error("Error exporting CSV:", error);
      toast.fromError(error, { description: <TranslatedMessage message="Failed to export CSV" /> });
    } finally {
      setExportLoading(false);
    }
  };

  const handleExport = async () => {
    if (exportType === "cloudzero") {
      // Check if settings exist, if not save them first
      if (!existingSettings) {
        let values: CloudZeroSettings | undefined;
        await form.handleSubmit((formValues) => {
          values = formValues;
        })();
        if (!values) return;
        const success = await handleSaveCloudZeroSettings(values);
        if (!success) return;
      }
      await handleExportCloudZero();
    } else {
      await handleExportCSV();
    }
  };

  const handleModalClose = () => {
    form.reset();
    setExportType("cloudzero");
    setExistingSettings(null);
    onClose();
  };

  const exportOptions = [
    {
      value: "cloudzero",
      label: (
        <div className="flex items-center gap-2">
          <img
            src="/cloudzero.png"
            alt="CloudZero"
            className="w-5 h-5"
            onError={(e) => {
              // Fallback to text if image fails to load
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
          <span>{t("Export to CloudZero")}</span>
        </div>
      ),
    },
    {
      value: "csv",
      label: (
        <div className="flex items-center gap-2">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <span>{t("Export to CSV")}</span>
        </div>
      ),
    },
  ];

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && handleModalClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>{t("Export Data")}</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          {/* Export Type Selection */}
          <div>
            <p className="text-sm font-medium mb-2 block">{t("Export Destination")}</p>
            <Select items={exportOptions} value={exportType} onValueChange={(value) => value && setExportType(value)}>
              <SelectTrigger className="w-full" aria-label={t("Export Destination")}>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {exportOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* CloudZero Configuration */}
          {exportType === "cloudzero" && (
            <div>
              {settingsLoading ? (
                <div className="flex justify-center py-8">
                  <UiLoadingSpinner className="size-8" />
                </div>
              ) : (
                <>
                  {existingSettings && (
                    <Alert className="mb-4">
                      <CircleCheck />
                      <AlertTitle>{t("Existing CloudZero Configuration")}</AlertTitle>
                      <AlertDescription>
                        {t("API Key:")} {existingSettings.api_key_masked}
                        <br />
                        {t("Connection ID:")} {existingSettings.connection_id}
                      </AlertDescription>
                    </Alert>
                  )}

                  {!existingSettings && (
                    <form onSubmit={(event) => event.preventDefault()}>
                      <FieldGroup>
                        <FormField
                          control={form.control}
                          name="api_key"
                          label={t("CloudZero API Key")}
                          renderError={({ message, type }) =>
                            type === "too_small" && message === "Please enter your CloudZero API key"
                              ? t(message)
                              : message
                          }
                        >
                          {({ ref, ...field }) => (
                            <PasswordInput {...field} ref={ref} placeholder={t("Enter your CloudZero API key")} />
                          )}
                        </FormField>

                        <FormField
                          control={form.control}
                          name="connection_id"
                          label={t("Connection ID")}
                          renderError={({ message, type }) =>
                            type === "too_small" && message === "Please enter the CloudZero connection ID"
                              ? t(message)
                              : message
                          }
                        >
                          {({ ref, ...field }) => (
                            <Input {...field} ref={ref} placeholder={t("Enter CloudZero connection ID")} />
                          )}
                        </FormField>
                      </FieldGroup>
                    </form>
                  )}
                </>
              )}
            </div>
          )}

          {/* CSV Export Info */}
          {exportType === "csv" && (
            <Alert variant="info">
              <FileDown />
              <AlertTitle>{t("CSV Export")}</AlertTitle>
              <AlertDescription>
                {t("Export your usage data as a CSV file for analysis in spreadsheet applications.")}
              </AlertDescription>
            </Alert>
          )}

          {/* Action Buttons */}
          <div className="flex justify-end space-x-2 pt-4">
            <Button type="button" variant="secondary" onClick={handleModalClose}>
              {t("Cancel")}
            </Button>
            <Button
              type="button"
              onClick={handleExport}
              disabled={loading || exportLoading}
              aria-busy={loading || exportLoading}
            >
              {(loading || exportLoading) && <UiLoadingSpinner className="size-4" />}
              {exportType === "cloudzero" ? t("Export to CloudZero") : t("Export CSV")}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default CloudZeroExportModal;
