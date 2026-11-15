import { useCloudZeroDryRun } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroDryRun";
import { useCloudZeroExport } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroExport";
import { useCloudZeroDeleteSettings } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroSettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Separator } from "@/components/ui/separator";
import { toast } from "@/lib/toast";
import { CheckCircle, Pencil, Play, Trash2, Upload } from "lucide-react";
import { useState } from "react";
import { useTranslation } from "react-i18next";
import CloudZeroUpdateModal from "./CloudZeroUpdateModal";
import { CloudZeroSettings } from "./types";

interface CloudZeroIntegrationSettingsProps {
  settings: CloudZeroSettings;
  onSettingsUpdated: () => void;
}

interface DetailRowProps {
  label: string;
  children: React.ReactNode;
}

const DetailRow = ({ label, children }: DetailRowProps) => (
  <div className="grid grid-cols-1 border-b border-border last:border-b-0 sm:grid-cols-[220px_minmax(0,1fr)]">
    <dt className="bg-muted/50 px-4 py-3 text-sm font-medium">{label}</dt>
    <dd className="px-4 py-3 text-sm">{children}</dd>
  </div>
);

const NotConfigured = () => {
  const { t } = useTranslation();
  return <span className="text-muted-foreground italic">{t("loggingAndAlerts.cloudZero.settings.notConfigured")}</span>;
};

export function CloudZeroIntegrationSettings({ settings, onSettingsUpdated }: CloudZeroIntegrationSettingsProps) {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [isExportConfirmOpen, setIsExportConfirmOpen] = useState(false);

  const dryRunMutation = useCloudZeroDryRun(accessToken || "");
  const exportMutation = useCloudZeroExport(accessToken || "");
  const deleteMutation = useCloudZeroDeleteSettings(accessToken || "");

  const handleDryRun = () => {
    if (!accessToken) return;

    dryRunMutation.mutate(
      { limit: 10 },
      {
        onSuccess: (data) => {
          toast.success(t("loggingAndAlerts.cloudZero.settings.toasts.dryRunSucceeded"));
        },
        onError: (error) => {
          toast.error(error?.message || t("loggingAndAlerts.cloudZero.settings.toasts.dryRunFailed"));
        },
      },
    );
  };

  const dryRunResult = dryRunMutation.data ? JSON.stringify(dryRunMutation.data, null, 2) : null;

  const handleExport = () => {
    if (!accessToken) return;

    exportMutation.mutate(
      { operation: "replace_hourly" },
      {
        onSuccess: () => {
          toast.success(t("loggingAndAlerts.cloudZero.settings.toasts.exported"));
          setIsExportConfirmOpen(false);
        },
        onError: (error) => {
          toast.error(error?.message || t("loggingAndAlerts.cloudZero.settings.toasts.exportFailed"));
        },
      },
    );
  };

  const handleEdit = () => {
    setIsEditModalOpen(true);
  };

  const handleEditModalOk = async () => {
    setIsEditModalOpen(false);
    onSettingsUpdated();
  };

  const handleEditModalCancel = () => {
    setIsEditModalOpen(false);
  };

  const handleDeleteClick = () => {
    setIsDeleteModalOpen(true);
  };

  const handleDeleteConfirm = () => {
    if (!accessToken) return;

    deleteMutation.mutate(undefined, {
      onSuccess: () => {
        toast.success(t("loggingAndAlerts.cloudZero.settings.toasts.deleted"));
        setIsDeleteModalOpen(false);
        onSettingsUpdated();
      },
      onError: (error) => {
        toast.error(error?.message || t("loggingAndAlerts.cloudZero.settings.toasts.deleteFailed"));
      },
    });
  };

  const handleDeleteCancel = () => {
    setIsDeleteModalOpen(false);
  };

  return (
    <>
      <div className="mx-auto w-full max-w-4xl space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              {t("loggingAndAlerts.cloudZero.settings.title")}
              <Badge variant="secondary" className="capitalize">
                {settings.status || t("loggingAndAlerts.cloudZero.settings.statusFallback")}
              </Badge>
            </CardTitle>
            <CardAction className="flex gap-2">
              <Button variant="outline" onClick={handleEdit}>
                <Pencil />
                {t("loggingAndAlerts.cloudZero.settings.edit")}
              </Button>
              <Button variant="destructive" onClick={handleDeleteClick}>
                <Trash2 />
                {t("loggingAndAlerts.cloudZero.settings.delete")}
              </Button>
            </CardAction>
          </CardHeader>

          <CardContent>
            <dl className="rounded-md border border-border">
              <DetailRow label={t("loggingAndAlerts.cloudZero.settings.fields.apiKey")}>
                <span className="font-mono">{settings.api_key_masked || <NotConfigured />}</span>
              </DetailRow>
              <DetailRow label={t("loggingAndAlerts.cloudZero.settings.fields.connectionId")}>
                <span className="font-mono">{settings.connection_id || <NotConfigured />}</span>
              </DetailRow>
              <DetailRow label={t("loggingAndAlerts.cloudZero.settings.fields.timezone")}>
                {settings.timezone || (
                  <span className="text-muted-foreground italic">
                    {t("loggingAndAlerts.cloudZero.settings.defaultTimezone")}
                  </span>
                )}
              </DetailRow>
            </dl>

            <div className="mt-6 flex items-center gap-3">
              <span className="text-sm text-muted-foreground">{t("loggingAndAlerts.cloudZero.settings.actions")}</span>
              <Separator className="flex-1" />
            </div>

            <div className="mt-4 mb-6 flex flex-wrap gap-4">
              <Button variant="outline" onClick={handleDryRun} disabled={dryRunMutation.isPending}>
                <Play />
                {t("loggingAndAlerts.cloudZero.settings.dryRun")}
              </Button>

              <Button onClick={() => setIsExportConfirmOpen(true)} disabled={exportMutation.isPending}>
                <Upload />
                {t("loggingAndAlerts.cloudZero.settings.exportNow")}
              </Button>
            </div>

            {dryRunResult && (
              <Alert>
                <CheckCircle />
                <AlertTitle>{t("loggingAndAlerts.cloudZero.settings.dryRunResults")}</AlertTitle>
                <AlertDescription>
                  <p>
                    {t("loggingAndAlerts.cloudZero.settings.simulationOutputPrefix")}
                    {settings.connection_id}
                  </p>
                  <pre className="overflow-x-auto rounded-md border border-border bg-muted p-4 font-mono text-xs text-foreground">
                    {dryRunResult}
                  </pre>
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      </div>

      <AlertDialog open={isExportConfirmOpen} onOpenChange={setIsExportConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("loggingAndAlerts.cloudZero.settings.exportDialog.title")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("loggingAndAlerts.cloudZero.settings.exportDialog.description")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={exportMutation.isPending}>
              {t("loggingAndAlerts.cloudZero.settings.exportDialog.cancel")}
            </AlertDialogCancel>
            <Button onClick={handleExport} disabled={exportMutation.isPending}>
              {t("loggingAndAlerts.cloudZero.settings.exportDialog.confirm")}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <CloudZeroUpdateModal
        open={isEditModalOpen}
        onOk={handleEditModalOk}
        onCancel={handleEditModalCancel}
        settings={settings}
      />

      <DeleteResourceModal
        isOpen={isDeleteModalOpen}
        title={t("loggingAndAlerts.cloudZero.settings.deleteDialog.title")}
        message={t("loggingAndAlerts.cloudZero.settings.deleteDialog.message")}
        resourceInformationTitle={t("loggingAndAlerts.cloudZero.settings.deleteDialog.infoTitle")}
        resourceInformation={[
          {
            label: t("loggingAndAlerts.cloudZero.settings.deleteDialog.connectionId"),
            value: settings.connection_id,
            code: true,
          },
          {
            label: t("loggingAndAlerts.cloudZero.settings.deleteDialog.timezone"),
            value: settings.timezone || t("loggingAndAlerts.cloudZero.settings.defaultTimezone"),
          },
        ]}
        onCancel={handleDeleteCancel}
        onOk={handleDeleteConfirm}
        confirmLoading={deleteMutation.isPending}
      />
    </>
  );
}
