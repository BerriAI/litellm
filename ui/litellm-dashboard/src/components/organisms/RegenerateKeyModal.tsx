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
import { Check, Copy, AlertTriangle, RefreshCw } from "lucide-react";
import { Form, InputNumber, Input as AntInput } from "antd";
import { add } from "date-fns";
import { useEffect, useState } from "react";
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

export function RegenerateKeyModal({
  selectedToken,
  visible,
  onClose,
  onKeyUpdate,
}: RegenerateKeyModalProps) {
  const { accessToken } = useAuthorized();
  const [form] = Form.useForm();
  const [regeneratedKey, setRegeneratedKey] = useState<string | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [regenerateFormData, setRegenerateFormData] = useState<any>(null);
  const [newExpiryTime, setNewExpiryTime] = useState<string | null>(null);
  const [isRegenerating, setIsRegenerating] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (visible && selectedToken && accessToken) {
      form.setFieldsValue({
        key_alias: selectedToken.key_alias,
        max_budget: selectedToken.max_budget,
        tpm_limit: selectedToken.tpm_limit,
        rpm_limit: selectedToken.rpm_limit,
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
      // Check "mo" before "m" to avoid a false prefix match (e.g. "1mo" → minutes).
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
    if (regenerateFormData?.duration) {
      setNewExpiryTime(calculateNewExpiryTime(regenerateFormData.duration));
    } else {
      setNewExpiryTime(null);
    }
  }, [regenerateFormData?.duration]);

  const handleRegenerateKey = async () => {
    if (!selectedToken || !accessToken) return;

    setIsRegenerating(true);
    try {
      const formValues = await form.validateFields();

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
        max_budget: formValues.max_budget,
        tpm_limit: formValues.tpm_limit,
        rpm_limit: formValues.rpm_limit,
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
  };

  const handleClose = () => {
    setRegeneratedKey(null);
    setIsRegenerating(false);
    setCopied(false);
    form.resetFields();
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
          <Form
            form={form}
            layout="vertical"
            className="mt-1"
            onValuesChange={(changedValues) => {
              if ("duration" in changedValues) {
                setRegenerateFormData(
                  (prev: { duration?: string }) => ({
                    ...prev,
                    duration: changedValues.duration,
                  }),
                );
              }
            }}
          >
            <Form.Item name="key_alias" label="Key Alias">
              <Input disabled />
            </Form.Item>

            <div className="grid grid-cols-3 gap-3">
              <Form.Item name="max_budget" label="Max Budget (USD)">
                <InputNumber
                  step={0.01}
                  precision={2}
                  style={{ width: "100%" }}
                />
              </Form.Item>
              <Form.Item name="tpm_limit" label="TPM Limit">
                <InputNumber style={{ width: "100%" }} />
              </Form.Item>
              <Form.Item name="rpm_limit" label="RPM Limit">
                <InputNumber style={{ width: "100%" }} />
              </Form.Item>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <Form.Item
                name="duration"
                label="Expire Key"
                extra={
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
                }
              >
                <AntInput placeholder="e.g. 30s, 30h, 30d" />
              </Form.Item>
              <Form.Item
                name="grace_period"
                label="Grace Period"
                tooltip="Keep the old key valid for this duration after rotation. Both keys work during this period for seamless cutover. Empty = immediate revoke."
                extra={
                  <span className="text-muted-foreground text-xs">
                    Recommended: 24h to 72h for production keys
                  </span>
                }
                rules={[
                  {
                    pattern: /^(\d+(s|m|h|d|w|mo))?$/,
                    message:
                      "Must be a duration like 30s, 30m, 24h, 2d, 1w, or 1mo",
                  },
                ]}
              >
                <AntInput placeholder="e.g. 24h, 2d" />
              </Form.Item>
            </div>
          </Form>
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
