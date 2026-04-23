import { useCloudZeroUpdateSettings } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroSettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import MessageManager from "@/components/molecules/message_manager";
import { useEffect, useState } from "react";
import { Eye, EyeOff } from "lucide-react";
import { CloudZeroSettings } from "./types";
import { FormProvider, useForm } from "react-hook-form";

interface CloudZeroUpdateModalProps {
  open: boolean;
  onOk: () => void;
  onCancel: () => void;
  settings: CloudZeroSettings;
}

interface UpdateValues {
  api_key: string;
  connection_id: string;
  timezone: string;
}

export default function CloudZeroUpdateModal({
  open,
  onOk,
  onCancel,
  settings,
}: CloudZeroUpdateModalProps) {
  const { accessToken } = useAuthorized();
  const updateMutation = useCloudZeroUpdateSettings(accessToken || "");
  const [showApiKey, setShowApiKey] = useState(false);
  const form = useForm<UpdateValues>({
    defaultValues: { api_key: "", connection_id: "", timezone: "" },
    mode: "onSubmit",
  });

  useEffect(() => {
    if (open && settings) {
      form.reset({
        connection_id: settings.connection_id ?? "",
        timezone: settings.timezone || "UTC",
        api_key: "",
      });
    } else if (open) {
      form.reset({ api_key: "", connection_id: "", timezone: "" });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, settings]);

  const handleSubmit = form.handleSubmit((values) => {
    updateMutation.mutate(
      {
        connection_id: values.connection_id,
        timezone: values.timezone || "UTC",
        ...(values.api_key && { api_key: values.api_key }),
      },
      {
        onSuccess: () => {
          MessageManager.success(
            "CloudZero integration updated successfully",
          );
          form.reset({ api_key: "", connection_id: "", timezone: "" });
          onOk();
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        onError: (error: any) => {
          MessageManager.error(
            error?.message || "Failed to update CloudZero integration",
          );
        },
      },
    );
  });

  const handleCancel = () => {
    form.reset({ api_key: "", connection_id: "", timezone: "" });
    onCancel();
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => (!o ? handleCancel() : undefined)}
    >
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Edit CloudZero Integration</DialogTitle>
          <DialogDescription className="sr-only">
            Update the CloudZero billing integration settings.
          </DialogDescription>
        </DialogHeader>
        <FormProvider {...form}>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="api_key">CloudZero API Key</Label>
              <div className="relative">
                <Input
                  id="api_key"
                  type={showApiKey ? "text" : "password"}
                  placeholder="Leave empty to keep existing"
                  {...form.register("api_key")}
                />
                <button
                  type="button"
                  onClick={() => setShowApiKey(!showApiKey)}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground"
                  aria-label={showApiKey ? "Hide api key" : "Show api key"}
                >
                  {showApiKey ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
              <p className="text-xs text-muted-foreground">
                Leave empty to keep the existing API key
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="connection_id">
                Connection ID <span className="text-destructive">*</span>
              </Label>
              <Input
                id="connection_id"
                placeholder="Enter your CloudZero connection ID"
                {...form.register("connection_id", {
                  required: "Please enter your CloudZero connection ID",
                })}
              />
              {form.formState.errors.connection_id && (
                <p className="text-sm text-destructive">
                  {form.formState.errors.connection_id.message as string}
                </p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="timezone">Timezone</Label>
              <Input
                id="timezone"
                placeholder="UTC"
                {...form.register("timezone")}
              />
              <p className="text-xs text-muted-foreground">
                Timezone for date handling (defaults to UTC if not provided)
              </p>
            </div>

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={handleCancel}
                disabled={updateMutation.isPending}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={updateMutation.isPending}>
                {updateMutation.isPending ? "Updating…" : "Update"}
              </Button>
            </DialogFooter>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
}
