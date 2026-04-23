import { useCloudZeroDryRun } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroDryRun";
import { useCloudZeroExport } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroExport";
import { useCloudZeroDeleteSettings } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroSettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import DeleteResourceModal from "@/components/common_components/DeleteResourceModal";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import MessageManager from "@/components/molecules/message_manager";
import { CheckCircle, Edit, Play, Trash2, Upload } from "lucide-react";
import { useState } from "react";
import CloudZeroUpdateModal from "./CloudZeroUpdateModal";
import { CloudZeroSettings } from "./types";

interface CloudZeroIntegrationSettingsProps {
  settings: CloudZeroSettings;
  onSettingsUpdated: () => void;
}

export function CloudZeroIntegrationSettings({
  settings,
  onSettingsUpdated,
}: CloudZeroIntegrationSettingsProps) {
  const { accessToken } = useAuthorized();
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);

  const dryRunMutation = useCloudZeroDryRun(accessToken || "");
  const exportMutation = useCloudZeroExport(accessToken || "");
  const deleteMutation = useCloudZeroDeleteSettings(accessToken || "");

  const handleDryRun = () => {
    if (!accessToken) return;
    dryRunMutation.mutate(
      { limit: 10 },
      {
        onSuccess: () => {
          MessageManager.success("Dry run completed successfully");
        },
        onError: (error) => {
          MessageManager.error(error?.message || "Failed to perform dry run");
        },
      },
    );
  };

  const dryRunResult = dryRunMutation.data
    ? JSON.stringify(dryRunMutation.data, null, 2)
    : null;

  const handleExport = () => {
    if (!accessToken) return;
    exportMutation.mutate(
      { operation: "replace_hourly" },
      {
        onSuccess: () => {
          MessageManager.success("Data successfully exported to CloudZero");
        },
        onError: (error) => {
          MessageManager.error(error?.message || "Failed to export data");
        },
      },
    );
  };

  const handleEditModalOk = async () => {
    setIsEditModalOpen(false);
    onSettingsUpdated();
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
        MessageManager.error(
          error?.message || "Failed to delete CloudZero integration",
        );
      },
    });
  };

  return (
    <>
      <div className="space-y-6 w-full max-w-4xl mx-auto">
        <Card className="shadow-sm p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <span className="text-lg font-semibold">
                CloudZero Configuration
              </span>
              <Badge variant="default" className="capitalize">
                {settings.status || "Active"}
              </Badge>
            </div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                onClick={() => setIsEditModalOpen(true)}
              >
                <Edit size={16} />
                Edit
              </Button>
              <Button
                variant="destructive"
                onClick={() => setIsDeleteModalOpen(true)}
              >
                <Trash2 size={16} />
                Delete
              </Button>
            </div>
          </div>

          <dl className="border border-border rounded-md overflow-hidden text-sm">
            <div className="grid grid-cols-[200px_1fr] border-b border-border">
              <dt className="bg-muted px-4 py-3 font-medium">
                API Key (Redacted)
              </dt>
              <dd className="px-4 py-3">
                <span className="font-mono text-muted-foreground">
                  {settings.api_key_masked || (
                    <span className="text-muted-foreground italic">
                      Not configured
                    </span>
                  )}
                </span>
              </dd>
            </div>
            <div className="grid grid-cols-[200px_1fr] border-b border-border">
              <dt className="bg-muted px-4 py-3 font-medium">Connection ID</dt>
              <dd className="px-4 py-3">
                <span className="font-mono text-muted-foreground">
                  {settings.connection_id || (
                    <span className="text-muted-foreground italic">
                      Not configured
                    </span>
                  )}
                </span>
              </dd>
            </div>
            <div className="grid grid-cols-[200px_1fr]">
              <dt className="bg-muted px-4 py-3 font-medium">Timezone</dt>
              <dd className="px-4 py-3">
                {settings.timezone || (
                  <span className="text-muted-foreground italic">
                    Default (UTC)
                  </span>
                )}
              </dd>
            </div>
          </dl>

          <Separator className="my-6" />
          <h4 className="text-sm font-medium text-muted-foreground mb-3">
            Actions
          </h4>

          <div className="flex flex-wrap gap-4 mb-6">
            <Button
              variant="outline"
              onClick={handleDryRun}
              disabled={dryRunMutation.isPending}
            >
              <Play size={16} />
              {dryRunMutation.isPending
                ? "Running…"
                : "Run Dry Run Simulation"}
            </Button>

            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button disabled={exportMutation.isPending}>
                  <Upload size={16} />
                  {exportMutation.isPending
                    ? "Exporting…"
                    : "Export Data Now"}
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>
                    Export Data to CloudZero
                  </AlertDialogTitle>
                  <AlertDialogDescription>
                    This will push the current accumulated cost data to
                    CloudZero. Continue?
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={handleExport}>
                    Export
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>

          {dryRunResult && (
            <div className="mt-6">
              <Alert>
                <CheckCircle className="h-4 w-4" />
                <AlertTitle>Dry Run Results</AlertTitle>
                <AlertDescription>
                  <p className="mb-2 text-muted-foreground">
                    Simulation output for connection: {settings.connection_id}
                  </p>
                  <pre className="bg-muted p-4 rounded-md border border-border overflow-x-auto text-xs font-mono text-foreground">
                    {dryRunResult}
                  </pre>
                </AlertDescription>
              </Alert>
            </div>
          )}
        </Card>
      </div>

      <CloudZeroUpdateModal
        open={isEditModalOpen}
        onOk={handleEditModalOk}
        onCancel={() => setIsEditModalOpen(false)}
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
        onCancel={() => setIsDeleteModalOpen(false)}
        onOk={handleDeleteConfirm}
        confirmLoading={deleteMutation.isPending}
      />
    </>
  );
}
