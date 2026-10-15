import { useState } from "react";
import { Pencil, Play, Trash2, Upload } from "lucide-react";

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

import { describeFrequency, formatEstimatedSpend } from "./ternaryDisplay";

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
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [isExportConfirmOpen, setIsExportConfirmOpen] = useState(false);
  const dryRun = useTernaryDryRun();
  const exportNow = useTernaryExport();
  const deleteSettings = useTernaryDelete();

  const handleDryRun = () =>
    dryRun.mutate(
      { limit: DRY_RUN_LIMIT },
      { onError: (error) => toast.error(error.message || "Failed to preview the export") },
    );

  const handleExport = () =>
    exportNow.mutate(
      {},
      {
        onSuccess: () => setIsExportConfirmOpen(false),
        onError: (error) => toast.error(error.message || "Failed to export to Ternary"),
      },
    );

  const handleDelete = () =>
    deleteSettings.mutate(undefined, {
      onSuccess: (response) => {
        setIsDeleteOpen(false);
        onDeleted(response.message);
      },
      onError: (error) => toast.error(error.message || "Failed to delete the Ternary integration"),
    });

  const summary = dryRun.data?.summary;

  return (
    <>
      <div className="mx-auto w-full max-w-4xl space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              Ternary Configuration
              <Badge variant="secondary" className="capitalize">
                {settings.status}
              </Badge>
            </CardTitle>
            <CardAction className="flex gap-2">
              <Button variant="outline" onClick={onEdit}>
                <Pencil />
                Edit
              </Button>
              <Button variant="destructive" onClick={() => setIsDeleteOpen(true)}>
                <Trash2 />
                Delete
              </Button>
            </CardAction>
          </CardHeader>

          <CardContent>
            <dl className="rounded-md border border-border">
              <DetailRow label="API Key (Redacted)">
                <span className="font-mono">{settings.api_key_masked}</span>
              </DetailRow>
              <DetailRow label="Connection ID">
                <span className="font-mono">{settings.connection_id}</span>
              </DetailRow>
              <DetailRow label="Base URL">
                <span className="font-mono">{settings.base_url}</span>
              </DetailRow>
              <DetailRow label="Export Frequency">{describeFrequency(settings)}</DetailRow>
            </dl>
            <p className="mt-3 text-xs text-muted-foreground">
              Settings saved here take effect for scheduled exports after the proxy restarts. A callbacks entry for
              ternary in the config file takes precedence over them.
            </p>

            <div className="mt-6 flex items-center gap-3">
              <span className="text-sm text-muted-foreground">Actions</span>
              <Separator className="flex-1" />
            </div>

            <div className="mt-4 mb-6 flex flex-wrap gap-4">
              <Button variant="outline" onClick={handleDryRun} disabled={dryRun.isPending}>
                <Play />
                Preview Export
              </Button>
              <Button onClick={() => setIsExportConfirmOpen(true)} disabled={exportNow.isPending}>
                <Upload />
                Export Now
              </Button>
            </div>

            {summary && (
              <Alert>
                <AlertTitle>Preview (LiteLLM estimated spend)</AlertTitle>
                <AlertDescription>
                  <p>
                    {summary.total_records} FOCUS rows across {summary.days.length} day(s), estimated spend{" "}
                    {formatEstimatedSpend(summary.estimated_spend)}, {summary.total_tokens.toLocaleString()} tokens,{" "}
                    {summary.unique_teams} team(s), {summary.unique_models} model(s). Nothing was sent to Ternary.
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Spend is LiteLLM&apos;s estimate, not your provider&apos;s invoice.
                  </p>
                  <pre className="max-h-96 overflow-auto rounded-md border border-border bg-muted p-4 font-mono text-xs text-foreground">
                    {JSON.stringify(dryRun.data?.normalized_data.slice(0, 5), null, 2)}
                  </pre>
                </AlertDescription>
              </Alert>
            )}

            {exportNow.data && (
              <Alert>
                <AlertTitle>Export complete</AlertTitle>
                <AlertDescription>
                  Uploaded usage from {exportNow.data.start_time_utc} to {exportNow.data.end_time_utc} (UTC), one
                  file per day.
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
      </div>

      <AlertDialog open={isExportConfirmOpen} onOpenChange={setIsExportConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Export to Ternary now?</AlertDialogTitle>
            <AlertDialogDescription>
              This sends the same days the next scheduled export would, replacing what Ternary already has for those
              days.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={exportNow.isPending}>Cancel</AlertDialogCancel>
            <Button onClick={handleExport} disabled={exportNow.isPending}>
              {exportNow.isPending ? "Exporting..." : "Export"}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <DeleteResourceModal
        isOpen={isDeleteOpen}
        title="Delete Ternary Integration?"
        message="This removes the stored Ternary settings. Scheduled exports stop after the proxy restarts."
        resourceInformationTitle="Integration Details"
        resourceInformation={[{ label: "Connection ID", value: settings.connection_id ?? "", code: true }]}
        onCancel={() => setIsDeleteOpen(false)}
        onOk={handleDelete}
        confirmLoading={deleteSettings.isPending}
      />
    </>
  );
}
