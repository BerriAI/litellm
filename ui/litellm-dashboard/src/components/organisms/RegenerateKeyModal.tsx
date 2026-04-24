import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
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
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Check,
  Copy,
  AlertTriangle,
  RefreshCw,
  HelpCircle,
} from "lucide-react";
import { add } from "date-fns";
import { useEffect, useState } from "react";
import { Controller, FormProvider, useForm, useWatch } from "react-hook-form";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { KeyResponse } from "../key_team_helpers/key_list";
import NotificationManager from "../molecules/notifications_manager";
import { regenerateKeyCall } from "../networking";

interface RegenerateKeyModalProps {
  selectedToken: KeyResponse | null;
  visible: boolean;
  onClose: () => void;
  onKeyUpdate?: (updatedKeyData: Partial<KeyResponse>) => void;
}

interface RegenerateFormValues {
  key_alias: string;
  max_budget: number | null;
  tpm_limit: number | null;
  rpm_limit: number | null;
  duration: string;
  grace_period: string;
}

export function RegenerateKeyModal({
  selectedToken,
  visible,
  onClose,
  onKeyUpdate,
}: RegenerateKeyModalProps) {
  const { accessToken } = useAuthorized();
  const form = useForm<RegenerateFormValues>({
    defaultValues: {
      key_alias: "",
      max_budget: null,
      tpm_limit: null,
      rpm_limit: null,
      duration: "",
      grace_period: "",
    },
  });
  const [regeneratedKey, setRegeneratedKey] = useState<string | null>(null);
  const [newExpiryTime, setNewExpiryTime] = useState<string | null>(null);
  const [isRegenerating, setIsRegenerating] = useState(false);
  const [copied, setCopied] = useState(false);

  const duration = useWatch({ control: form.control, name: "duration" });

  useEffect(() => {
    if (visible && selectedToken && accessToken) {
      form.reset({
        key_alias: selectedToken.key_alias ?? "",
        max_budget:
          (selectedToken.max_budget as number | null | undefined) ?? null,
        tpm_limit:
          (selectedToken.tpm_limit as number | null | undefined) ?? null,
        rpm_limit:
          (selectedToken.rpm_limit as number | null | undefined) ?? null,
        duration: selectedToken.duration || "",
        grace_period: "",
      });
    }
  }, [visible, selectedToken, form, accessToken]);

  const calculateNewExpiryTime = (
    duration: string | undefined,
  ): string | null => {
    if (!duration) return null;

    try {
      const amount = parseInt(duration);
      if (Number.isNaN(amount)) {
        throw new Error("Invalid duration format");
      }
      const now = new Date();
      let newExpiry: Date;
      if (duration.endsWith("mo")) {
        newExpiry = add(now, { months: amount });
      } else if (duration.endsWith("s")) {
        newExpiry = add(now, { seconds: amount });
      } else if (duration.endsWith("m")) {
        newExpiry = add(now, { minutes: amount });
      } else if (duration.endsWith("h")) {
        newExpiry = add(now, { hours: amount });
      } else if (duration.endsWith("d")) {
        newExpiry = add(now, { days: amount });
      } else if (duration.endsWith("w")) {
        newExpiry = add(now, { weeks: amount });
      } else {
        throw new Error("Invalid duration format");
      }

      return newExpiry.toLocaleString();
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (_error) {
      return null;
    }
  };

  useEffect(() => {
    if (duration) {
      setNewExpiryTime(calculateNewExpiryTime(duration));
    } else {
      setNewExpiryTime(null);
    }
  }, [duration]);

  const handleRegenerateKey = form.handleSubmit(async (formValues) => {
    if (!selectedToken || !accessToken) return;

    setIsRegenerating(true);
    try {
      const response = await regenerateKeyCall(
        accessToken,
        selectedToken.token || selectedToken.token_id,
        formValues,
      );
      setRegeneratedKey(response.key);
      NotificationManager.success("Virtual Key regenerated successfully");

      const updatedKeyData: Partial<KeyResponse> = {
        ...response,
        token: response.token || response.key_id || selectedToken.token,
        key_name: response.key,
        max_budget: formValues.max_budget as number | undefined,
        tpm_limit: formValues.tpm_limit as number | undefined,
        rpm_limit: formValues.rpm_limit as number | undefined,
        expires: formValues.duration
          ? (calculateNewExpiryTime(formValues.duration) ??
            selectedToken.expires)
          : selectedToken.expires,
      };

      if (onKeyUpdate) {
        onKeyUpdate(updatedKeyData);
      }

      setIsRegenerating(false);
    } catch (error) {
      console.error("Error regenerating key:", error);
      NotificationManager.fromBackend(error);
      setIsRegenerating(false);
    }
  });

  const handleClose = () => {
    setRegeneratedKey(null);
    setIsRegenerating(false);
    setCopied(false);
    form.reset({
      key_alias: "",
      max_budget: null,
      tpm_limit: null,
      rpm_limit: null,
      duration: "",
      grace_period: "",
    });
    onClose();
  };

  const handleCopyKey = () => {
    setCopied(true);
  };

  return (
    <Dialog open={visible} onOpenChange={(o) => (!o ? handleClose() : undefined)}>
      <DialogContent
        className="max-w-[520px]"
        onInteractOutside={(e) => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle>Regenerate Virtual Key</DialogTitle>
        </DialogHeader>

        {regeneratedKey ? (
          <div className="flex flex-col gap-4">
            {/* eslint-disable-next-line litellm-ui/no-raw-tailwind-colors */}
            <div className="flex gap-2 items-start p-3 rounded-md bg-amber-50 border border-amber-200 text-amber-800 dark:bg-amber-950/30 dark:border-amber-900 dark:text-amber-200">
              <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
              <span className="text-sm">
                Save it now, you will not see it again
              </span>
            </div>

            <div className="flex flex-col gap-0.5">
              <span className="text-muted-foreground text-xs">Key Alias</span>
              <span>{selectedToken?.key_alias || "No alias set"}</span>
            </div>

            <div className="flex flex-col gap-1.5">
              <span className="text-muted-foreground text-xs">
                Virtual Key
              </span>
              <div className="bg-muted border border-border rounded-md px-4 py-3.5 font-mono text-base break-all">
                {regeneratedKey}
              </div>
            </div>
          </div>
        ) : (
          <FormProvider {...form}>
            <form onSubmit={handleRegenerateKey} className="mt-1 space-y-4">
              <div className="space-y-2">
                <Label htmlFor="key_alias">Key Alias</Label>
                <Input id="key_alias" disabled {...form.register("key_alias")} />
              </div>

              <div className="grid grid-cols-3 gap-3">
                <div className="space-y-2">
                  <Label htmlFor="max_budget">Max Budget (USD)</Label>
                  <Controller
                    control={form.control}
                    name="max_budget"
                    render={({ field }) => (
                      <Input
                        id="max_budget"
                        type="number"
                        step={0.01}
                        value={field.value ?? ""}
                        onChange={(e) => {
                          const v = e.target.value;
                          field.onChange(v === "" ? null : Number(v));
                        }}
                      />
                    )}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="tpm_limit">TPM Limit</Label>
                  <Controller
                    control={form.control}
                    name="tpm_limit"
                    render={({ field }) => (
                      <Input
                        id="tpm_limit"
                        type="number"
                        value={field.value ?? ""}
                        onChange={(e) => {
                          const v = e.target.value;
                          field.onChange(v === "" ? null : Number(v));
                        }}
                      />
                    )}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="rpm_limit">RPM Limit</Label>
                  <Controller
                    control={form.control}
                    name="rpm_limit"
                    render={({ field }) => (
                      <Input
                        id="rpm_limit"
                        type="number"
                        value={field.value ?? ""}
                        onChange={(e) => {
                          const v = e.target.value;
                          field.onChange(v === "" ? null : Number(v));
                        }}
                      />
                    )}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-2">
                  <Label htmlFor="duration">Expire Key</Label>
                  <Input
                    id="duration"
                    placeholder="e.g. 30s, 30h, 30d"
                    {...form.register("duration")}
                  />
                  <div className="flex flex-col gap-0.5">
                    <span className="text-muted-foreground text-xs">
                      Current expiry:{" "}
                      {selectedToken?.expires
                        ? new Date(selectedToken.expires).toLocaleString()
                        : "Never"}
                    </span>
                    {newExpiryTime && (
                      // eslint-disable-next-line litellm-ui/no-raw-tailwind-colors
                      <span className="text-emerald-600 text-xs">
                        New expiry: {newExpiryTime}
                      </span>
                    )}
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center gap-1">
                    <Label htmlFor="grace_period">Grace Period</Label>
                    <TooltipProvider>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <HelpCircle className="h-3.5 w-3.5 text-muted-foreground" />
                        </TooltipTrigger>
                        <TooltipContent>
                          Keep the old key valid for this duration after rotation.
                          Both keys work during this period for seamless cutover.
                          Empty = immediate revoke.
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  </div>
                  <Input
                    id="grace_period"
                    placeholder="e.g. 24h, 2d"
                    aria-invalid={!!form.formState.errors.grace_period}
                    {...form.register("grace_period", {
                      pattern: {
                        value: /^(\d+(s|m|h|d|w|mo))?$/,
                        message:
                          "Must be a duration like 30s, 30m, 24h, 2d, 1w, or 1mo",
                      },
                    })}
                  />
                  {form.formState.errors.grace_period ? (
                    <p className="text-sm text-destructive">
                      {form.formState.errors.grace_period.message as string}
                    </p>
                  ) : (
                    <span className="text-muted-foreground text-xs">
                      Recommended: 24h to 72h for production keys
                    </span>
                  )}
                </div>
              </div>
            </form>
          </FormProvider>
        )}

        <DialogFooter>
          {regeneratedKey ? (
            <>
              <Button variant="outline" onClick={handleClose}>
                Close
              </Button>
              <CopyToClipboard text={regeneratedKey} onCopy={handleCopyKey}>
                <Button>
                  {copied ? (
                    <Check className="h-4 w-4" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                  {copied ? "Copied" : "Copy Key"}
                </Button>
              </CopyToClipboard>
            </>
          ) : (
            <>
              <Button variant="outline" onClick={handleClose}>
                Cancel
              </Button>
              <Button
                onClick={handleRegenerateKey}
                disabled={isRegenerating}
              >
                <RefreshCw
                  className={
                    isRegenerating ? "h-4 w-4 animate-spin" : "h-4 w-4"
                  }
                />
                Regenerate
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
