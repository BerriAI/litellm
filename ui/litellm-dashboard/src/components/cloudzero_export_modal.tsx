import React, { useState, useEffect } from "react";
import { FileText, CheckCircle2, Plus } from "lucide-react";
import { Controller, FormProvider, useForm } from "react-hook-form";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { getGlobalLitellmHeaderName } from "@/components/networking";
import NotificationsManager from "./molecules/notifications_manager";

interface CloudZeroExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  accessToken: string | null;
}

interface CloudZeroSettings {
  api_key: string;
  connection_id: string;
}

interface CloudZeroSettingsView {
  api_key_masked: string;
  connection_id: string;
  status: string;
}

type ExportType = "cloudzero" | "csv";

const CloudZeroExportModal: React.FC<CloudZeroExportModalProps> = ({ isOpen, onClose, accessToken }) => {
  const form = useForm<CloudZeroSettings>({
    defaultValues: { api_key: "", connection_id: "" },
  });
  const [loading, setLoading] = useState(false);
  const [existingSettings, setExistingSettings] = useState<CloudZeroSettingsView | null>(null);
  const [settingsLoading, setSettingsLoading] = useState(false);
  const [exportType, setExportType] = useState<ExportType>("cloudzero");
  const [exportLoading, setExportLoading] = useState(false);

  useEffect(() => {
    if (isOpen && accessToken) {
      loadExistingSettings();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
        form.reset({
          api_key: "",
          connection_id: settings.connection_id,
        });
      } else if (response.status !== 404) {
        const errorData = await response.json();
        NotificationsManager.fromBackend(`Failed to load existing settings: ${errorData.error || "Unknown error"}`);
      }
    } catch (error) {
      console.error("Error loading CloudZero settings:", error);
      NotificationsManager.fromBackend("Failed to load existing settings");
    } finally {
      setSettingsLoading(false);
    }
  };

  const handleSaveCloudZeroSettings = async (values: CloudZeroSettings) => {
    if (!accessToken) {
      NotificationsManager.fromBackend("No access token available");
      return;
    }

    setLoading(true);
    try {
      const endpoint = existingSettings ? "/cloudzero/settings" : "/cloudzero/init";
      const method = existingSettings ? "PUT" : "POST";

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

      const data = await response.json();

      if (response.ok) {
        NotificationsManager.success(data.message || "CloudZero settings saved successfully");
        setExistingSettings({
          api_key_masked: values.api_key.substring(0, 4) + "****" + values.api_key.slice(-4),
          connection_id: values.connection_id,
          status: "configured",
        });
        return true;
      } else {
        NotificationsManager.fromBackend(data.error || "Failed to save CloudZero settings");
        return false;
      }
    } catch (error) {
      console.error("Error saving CloudZero settings:", error);
      NotificationsManager.fromBackend("Failed to save CloudZero settings");
      return false;
    } finally {
      setLoading(false);
    }
  };

  const handleExportCloudZero = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("No access token available");
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

      const data = await response.json();

      if (response.ok) {
        NotificationsManager.success(data.message || "Export to CloudZero completed successfully");
        onClose();
      } else {
        NotificationsManager.fromBackend(data.error || "Failed to export to CloudZero");
      }
    } catch (error) {
      console.error("Error exporting to CloudZero:", error);
      NotificationsManager.fromBackend("Failed to export to CloudZero");
    } finally {
      setExportLoading(false);
    }
  };

  const handleExportCSV = async () => {
    setExportLoading(true);
    try {
      NotificationsManager.info("CSV export functionality coming soon!");
      onClose();
    } catch (error) {
      console.error("Error exporting CSV:", error);
      NotificationsManager.fromBackend("Failed to export CSV");
    } finally {
      setExportLoading(false);
    }
  };

  const handleExport = async () => {
    if (exportType === "cloudzero") {
      if (!existingSettings) {
        const ok = await form.trigger();
        if (!ok) return;
        const values = form.getValues();
        const success = await handleSaveCloudZeroSettings(values);
        if (!success) return;
      }
      await handleExportCloudZero();
    } else {
      await handleExportCSV();
    }
  };

  const handleModalClose = (open: boolean) => {
    if (open) return;
    form.reset({ api_key: "", connection_id: "" });
    setExportType("cloudzero");
    setExistingSettings(null);
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleModalClose}>
      <DialogContent className="max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Export Data</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          {/* Export Type Selection */}
          <div className="space-y-2">
            <Label className="font-medium">Export Destination</Label>
            <Select value={exportType} onValueChange={(v) => setExportType(v as ExportType)}>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="cloudzero">
                  <div className="flex items-center gap-2">
                    <img
                      src="/cloudzero.png"
                      alt="CloudZero"
                      className="w-5 h-5"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = "none";
                      }}
                    />
                    <span>Export to CloudZero</span>
                  </div>
                </SelectItem>
                <SelectItem value="csv">
                  <div className="flex items-center gap-2">
                    <FileText className="w-5 h-5" />
                    <span>Export to CSV</span>
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* CloudZero Configuration */}
          {exportType === "cloudzero" && (
            <div>
              {settingsLoading ? (
                <div className="flex justify-center py-8">
                  <Skeleton className="h-10 w-full" />
                </div>
              ) : (
                <>
                  {existingSettings && (
                    <div className="mb-4 rounded-md border border-border bg-muted p-4 flex gap-3">
                      <CheckCircle2 className="h-5 w-5 mt-0.5 text-primary shrink-0" />
                      <div className="space-y-1">
                        <p className="font-medium">Existing CloudZero Configuration</p>
                        <p className="text-sm text-muted-foreground">
                          API Key: {existingSettings.api_key_masked}
                          <br />
                          Connection ID: {existingSettings.connection_id}
                        </p>
                      </div>
                    </div>
                  )}

                  {!existingSettings && (
                    <FormProvider {...form}>
                      <form className="space-y-4">
                        <div className="space-y-2">
                          <Label htmlFor="cz-api-key">
                            CloudZero API Key <span className="text-destructive">*</span>
                          </Label>
                          <Controller
                            control={form.control}
                            name="api_key"
                            rules={{ required: "Please enter your CloudZero API key" }}
                            render={({ field, fieldState }) => (
                              <>
                                <Input
                                  id="cz-api-key"
                                  type="password"
                                  placeholder="Enter your CloudZero API key"
                                  {...field}
                                />
                                {fieldState.error && (
                                  <p className="text-sm text-destructive">{fieldState.error.message}</p>
                                )}
                              </>
                            )}
                          />
                        </div>

                        <div className="space-y-2">
                          <Label htmlFor="cz-conn-id">
                            Connection ID <span className="text-destructive">*</span>
                          </Label>
                          <Controller
                            control={form.control}
                            name="connection_id"
                            rules={{ required: "Please enter the CloudZero connection ID" }}
                            render={({ field, fieldState }) => (
                              <>
                                <Input
                                  id="cz-conn-id"
                                  placeholder="Enter CloudZero connection ID"
                                  {...field}
                                />
                                {fieldState.error && (
                                  <p className="text-sm text-destructive">{fieldState.error.message}</p>
                                )}
                              </>
                            )}
                          />
                        </div>
                      </form>
                    </FormProvider>
                  )}
                </>
              )}
            </div>
          )}

          {/* CSV Export Info */}
          {exportType === "csv" && (
            <div className="rounded-md border border-border bg-muted p-4 flex gap-3">
              <Plus className="h-5 w-5 mt-0.5 text-primary shrink-0" />
              <div>
                <p className="font-medium">CSV Export</p>
                <p className="text-sm text-muted-foreground">
                  Export your usage data as a CSV file for analysis in spreadsheet applications.
                </p>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex justify-end space-x-2 pt-4">
            <Button variant="outline" onClick={() => handleModalClose(false)}>
              Cancel
            </Button>
            <Button onClick={handleExport} disabled={loading || exportLoading}>
              {exportType === "cloudzero" ? "Export to CloudZero" : "Export CSV"}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default CloudZeroExportModal;
