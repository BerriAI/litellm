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
import MessageManager from "@/components/molecules/message_manager";
import { CheckCircle, Pencil, Play, Trash2, Upload } from "lucide-react";
import { useState } from "react";
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

const NotConfigured = () => <span className="text-muted-foreground italic">Not configured</span>;

export function CloudZeroIntegrationSettings({ settings, onSettingsUpdated }: CloudZeroIntegrationSettingsProps) {
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
          MessageManager.success("Dry run completed successfully");
        },
        onError: (error) => {
          MessageManager.error(error?.message || "Failed to perform dry run");
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
          MessageManager.success("Data successfully exported to CloudZero");
          setIsExportConfirmOpen(false);
        },
        onError: (error) => {
          MessageManager.error(error?.message || "Failed to export data");
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
        MessageManager.success("CloudZero integration deleted successfully");
        setIsDeleteModalOpen(false);
        onSettingsUpdated();
      },
      onError: (error) => {
        MessageManager.error(error?.message || "Failed to delete CloudZero integration");
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
              CloudZero Configuration
              <Badge variant="secondary" className="capitalize">
                {settings.status || "Active"}
              </Badge>
            </CardTitle>
            <CardAction className="flex gap-2">
              <Button variant="outline" onClick={handleEdit}>
                <Pencil />
                Edit
              </Button>
              <Button variant="destructive" onClick={handleDeleteClick}>
                <Trash2 />
                Delete
              </Button>
            </CardAction>
          </CardHeader>

          <CardContent>
            <dl className="rounded-md border border-border">
              <DetailRow label="API Key (Redacted)">
                <span className="font-mono">{settings.api_key_masked || <NotConfigured />}</span>
              </DetailRow>
              <DetailRow label="Connection ID">
                <span className="font-mono">{settings.connection_id || <NotConfigured />}</span>
              </DetailRow>
              <DetailRow label="Timezone">
                {settings.timezone || <span className="text-muted-foreground italic">Default (UTC)</span>}
              </DetailRow>
            </dl>

            <div className="mt-6 flex items-center gap-3">
              <span className="text-sm text-muted-foreground">Actions</span>
              <Separator className="flex-1" />
            </div>

            <div className="mt-4 mb-6 flex flex-wrap gap-4">
              <Button variant="outline" onClick={handleDryRun} disabled={dryRunMutation.isPending}>
                <Play />
                Run Dry Run Simulation
              </Button>

              <Button onClick={() => setIsExportConfirmOpen(true)} disabled={exportMutation.isPending}>
                <Upload />
                Export Data Now
              </Button>
            </div>

            {dryRunResult && (
              <Alert>
                <CheckCircle />
                <AlertTitle>Dry Run Results</AlertTitle>
                <AlertDescription>
                  <p>Simulation output for connection: {settings.connection_id}</p>
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
            <AlertDialogTitle>Export Data to CloudZero</AlertDialogTitle>
            <AlertDialogDescription>
              This will push the current accumulated cost data to CloudZero. Continue?
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={exportMutation.isPending}>Cancel</AlertDialogCancel>
            <Button onClick={handleExport} disabled={exportMutation.isPending}>
              Export
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
        title="Delete CloudZero Integration?"
        message="Are you sure you want to delete this CloudZero integration? All associated settings and configurations will be permanently removed."
        resourceInformationTitle="Integration Details"
        resourceInformation={[
          {
            label: "Connection ID",
            value: settings.connection_id,
            code: true,
          },
          {
            label: "Timezone",
            value: settings.timezone || "Default (UTC)",
          },
        ]}
        onCancel={handleDeleteCancel}
        onOk={handleDeleteConfirm}
        confirmLoading={deleteMutation.isPending}
      />
    </>
  );
}
