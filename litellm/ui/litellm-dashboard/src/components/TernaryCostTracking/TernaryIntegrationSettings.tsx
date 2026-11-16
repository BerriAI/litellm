import { useState } from "react";
import { Pencil, Play, Trash2, Upload } from "lucide-react";
import { useTranslation } from "react-i18next";

import {
  useTernaryDelete,
  useTernaryDryRun,
  useTernaryExport,
  type TernarySettingsView,
} from "@/app/(dashboard)/hooks/ternary/useTernary";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { toast } from "@/lib/toast";

import { describeExport, describeFrequency, describePreview } from "./ternaryDisplay";

const DRY_RUN_LIMIT = 500;

interface TernaryIntegrationSettingsProps {
  settings: TernarySettingsView;
  onEdit: () => void;
  onDeleted: (message: string) => void;
}

const DetailRow = ({ label, children }: { label: string; children: React.ReactNode }) => (
  <div className="grid grid-cols-1 border-b border-border last:border-b-0 sm:grid-cols-[220px_minmax(0,1fr)]">
    <dt className="bg-muted/50 px-4 py-3 text-sm font-medium">{label}</dt>
    <dd className="px-4 py-3 text-sm">{children}</dd>
  </div>
);

export function TernaryIntegrationSettings({ settings, onEdit, onDeleted }: TernaryIntegrationSettingsProps) {
  const { t } = useTranslation();
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [isExportConfirmOpen, setIsExportConfirmOpen] = useState(false);
  const dryRun = useTernaryDryRun();
  const exportNow = useTernaryExport();
  const deleteSettings = useTernaryDelete();

  const handleDryRun = () =>
    dryRun.mutate(
      { limit: DRY_RUN_LIMIT },
      { onError: (error) => toast.error(error.message || t("loggingAndAlerts.ternary.settings.toasts.previewFailed")) },
    );

  const handleExport = () =>
    exportNow.mutate(
      {},
      {
        onSuccess: () => setIsExportConfirmOpen(false),
        onError: (error) => toast.error(error.message || t("loggingAndAlerts.ternary.settings.toasts.exportFailed")),
      },
    );

  const handleDelete = () =>
    deleteSettings.mutate(undefined, {
      onSuccess: (response) => {
        setIsDeleteOpen(false);
        onDeleted(response.message);
      },
      onError: (error) => toast.error(error.message || t("loggingAndAlerts.ternary.settings.toasts.deleteFailed")),
    });

  const summary = dryRun.data?.summary;
  const managedByConfigFile = settings.status === "config_file";

  return (
    <>
      <div className="mx-auto w-full max-w-4xl space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              {t("loggingAndAlerts.ternary.settings.title")}
              <Badge variant="secondary">
                {managedByConfigFile
                  ? t("loggingAndAlerts.ternary.settings.statusConfigFile")
                  : t("loggingAndAlerts.ternary.settings.statusConfigured")}
              </Badge>
            </CardTitle>
            {!managedByConfigFile && (
              <CardAction className="flex gap-2">
                <Button variant="outline" onClick={onEdit}>
                  <Pencil />
                  {t("loggingAndAlerts.ternary.settings.edit")}
                </Button>
                <Button variant="destructive" onClick={() => setIsDeleteOpen(true)}>
                  <Trash2 />
                  {t("loggingAndAlerts.ternary.settings.delete")}
                </Button>
              </CardAction>
            )}
          </CardHeader>

          <CardContent>
            {managedByConfigFile ? (
              <p className="text-sm text-muted-foreground">{t("loggingAndAlerts.ternary.settings.configFileNotice")}</p>
            ) : (
              <>
                <dl className="rounded-md border border-border">
                  <DetailRow label={t("loggingAndAlerts.ternary.settings.fields.apiKey")}>
                    <span className="font-mono">{settings.api_key_masked}</span>
                  </DetailRow>
                  <DetailRow label={t("loggingAndAlerts.ternary.settings.fields.connectionId")}>
                    <span className="font-mono">{settings.connection_id}</span>
                  </DetailRow>
                  <DetailRow label={t("loggingAndAlerts.ternary.settings.fields.baseUrl")}>
                    <span className="font-mono">{settings.ternary_base_url}</span>
                  </DetailRow>
                  <DetailRow label={t("loggingAndAlerts.ternary.settings.fields.frequency")}>
                    {describeFrequency(settings)}
                  </DetailRow>
                </dl>
                <p className="mt-3 text-xs text-muted-foreground">
                  {t("loggingAndAlerts.ternary.settings.restartHint")}
                </p>
              </>
            )}

            <div className="mt-6 flex items-center gap-3">
              <span className="text-sm text-muted-foreground">{t("loggingAndAlerts.ternary.settings.actions")}</span>
              <Separator className="flex-1" />
            </div>

            <div className="mt-4 mb-6 flex flex-wrap gap-4">
              <Button variant="outline" onClick={handleDryRun} disabled={dryRun.isPending}>
                <Play />
                {t("loggingAndAlerts.ternary.settings.preview")}
              </Button>
              <Button onClick={() => setIsExportConfirmOpen(true)} disabled={exportNow.isPending}>
                <Upload />
                {t("loggingAndAlerts.ternary.settings.exportNow")}
              </Button>
            </div>

            {summary && (
              <Alert>
                <AlertTitle>{t("loggingAndAlerts.ternary.settings.previewTitle")}</AlertTitle>
                <AlertDescription className="min-w-0">
                  <p>{describePreview(summary)}</p>
                  <p className="text-xs text-muted-foreground">
                    {t("loggingAndAlerts.ternary.settings.previewEstimateNote")}
                  </p>
                  <pre className="max-h-96 w-full max-w-full overflow-auto rounded-md border border-border bg-muted p-4 font-mono text-xs text-foreground">
                    {JSON.stringify(dryRun.data?.normalized_data.slice(0, 5), null, 2)}
                  </pre>
                </AlertDescription>
              </Alert>
            )}

            {exportNow.data && (
              <Alert>
                <AlertTitle>
                  {exportNow.data.days_uploaded.length
                    ? t("loggingAndAlerts.ternary.settings.exportComplete")
                    : t("loggingAndAlerts.ternary.settings.nothingToExport")}
                </AlertTitle>
                <AlertDescription>{describeExport(exportNow.data)}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      </div>

      <AlertDialog open={isExportConfirmOpen} onOpenChange={setIsExportConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("loggingAndAlerts.ternary.settings.exportDialog.title")}</AlertDialogTitle>
            <AlertDialogDescription>
              {t("loggingAndAlerts.ternary.settings.exportDialog.description")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={exportNow.isPending}>
              {t("loggingAndAlerts.ternary.settings.exportDialog.cancel")}
            </AlertDialogCancel>
            <Button onClick={handleExport} disabled={exportNow.isPending}>
              {exportNow.isPending
                ? t("loggingAndAlerts.ternary.settings.exportDialog.exporting")
                : t("loggingAndAlerts.ternary.settings.exportDialog.confirm")}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <DeleteResourceModal
        isOpen={isDeleteOpen}
        title={t("loggingAndAlerts.ternary.settings.deleteDialog.title")}
        message={t("loggingAndAlerts.ternary.settings.deleteDialog.message")}
        resourceInformationTitle={t("loggingAndAlerts.ternary.settings.deleteDialog.infoTitle")}
        resourceInformation={[
          {
            label: t("loggingAndAlerts.ternary.settings.deleteDialog.connectionId"),
            value: settings.connection_id ?? "",
            code: true,
          },
        ]}
        onCancel={() => setIsDeleteOpen(false)}
        onOk={handleDelete}
        confirmLoading={deleteSettings.isPending}
      />
    </>
  );
}
