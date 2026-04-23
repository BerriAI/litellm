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
import { useEffect } from "react";
import { Eye, EyeOff } from "lucide-react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useCloudZeroCreate } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroCreate";
import { FormProvider, useForm } from "react-hook-form";
import { useState } from "react";

interface CloudZeroCreationModalProps {
  open: boolean;
  onOk: () => void;
  onCancel: () => void;
}

interface CreateValues {
  api_key: string;
  connection_id: string;
  timezone: string;
}

const defaultValues: CreateValues = {
  api_key: "",
  connection_id: "",
  timezone: "",
};

export default function CloudZeroCreationModal({
  open,
  onOk,
  onCancel,
}: CloudZeroCreationModalProps) {
  const { accessToken } = useAuthorized();
  const form = useForm<CreateValues>({
    defaultValues,
    mode: "onSubmit",
  });
  const createMutation = useCloudZeroCreate(accessToken || "");
  const [showApiKey, setShowApiKey] = useState(false);

  useEffect(() => {
    if (open) form.reset(defaultValues);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  const handleSubmit = form.handleSubmit((values) => {
    createMutation.mutate(
      {
        connection_id: values.connection_id,
        timezone: values.timezone || "UTC",
        ...(values.api_key && { api_key: values.api_key }),
      },
      {
        onSuccess: () => {
          MessageManager.success(
            "CloudZero integration created successfully",
          );
          form.reset(defaultValues);
          onOk();
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        onError: (error: any) => {
          MessageManager.error(
            error?.message || "Failed to create CloudZero integration",
          );
        },
      },
    );
  });

  const handleCancel = () => {
    form.reset(defaultValues);
    onCancel();
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(o) => (!o ? handleCancel() : undefined)}
    >
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Create CloudZero Integration</DialogTitle>
          <DialogDescription className="sr-only">
            Configure a new CloudZero billing integration.
          </DialogDescription>
        </DialogHeader>
        <FormProvider {...form}>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="api_key">
                CloudZero API Key <span className="text-destructive">*</span>
              </Label>
              <div className="relative">
                <Input
                  id="api_key"
                  type={showApiKey ? "text" : "password"}
                  placeholder="Enter your CloudZero API key"
                  {...form.register("api_key", {
                    required: "Please enter your CloudZero API key",
                  })}
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
              {form.formState.errors.api_key && (
                <p className="text-sm text-destructive">
                  {form.formState.errors.api_key.message as string}
                </p>
              )}
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
                disabled={createMutation.isPending}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={createMutation.isPending}>
                {createMutation.isPending ? "Creating…" : "Create"}
              </Button>
            </DialogFooter>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
}
