import { useLanguage } from "@/contexts/LanguageContext";
import { CircleAlert } from "lucide-react";
import React, { useState } from "react";
import { Alert, AlertDescription, AlertTitle } from "@/components/shared/Alert";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { toast } from "@/lib/toast";
import { AgentKillSwitchResult, triggerAgentKillSwitchCall } from "@/components/networking";
import { KillSwitchConfig } from "./kill_switch_config";

interface AgentKillSwitchDangerZoneProps {
  agentId: string;
  agentName: string;
  killSwitch: KillSwitchConfig | null | undefined;
  accessToken: string | null;
  isAdmin: boolean;
}

const AgentKillSwitchDangerZone: React.FC<AgentKillSwitchDangerZoneProps> = ({
  agentId,
  agentName,
  killSwitch,
  accessToken,
  isAdmin,
}) => {
  const { t } = useLanguage();
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const [confirmationInput, setConfirmationInput] = useState("");
  const [isFiring, setIsFiring] = useState(false);
  const [lastResult, setLastResult] = useState<AgentKillSwitchResult | null>(null);

  if (!isAdmin) return null;

  const openConfirm = () => {
    setConfirmationInput("");
    setIsConfirmOpen(true);
  };

  const fire = async () => {
    if (!accessToken) return;
    setIsFiring(true);
    setLastResult(null);
    try {
      const result = await triggerAgentKillSwitchCall(accessToken, agentId);
      setLastResult(result);
      setIsConfirmOpen(false);
      toast.success(t("Kill switch fired (HTTP {status_code})", { status_code: String(result.status_code) }));
    } catch (error) {
      toast.error(error instanceof Error ? error.message : t("Failed to fire kill switch"));
    } finally {
      setIsFiring(false);
    }
  };

  return (
    <section aria-labelledby="agent-danger-zone-heading" className="mt-6">
      <h3 id="agent-danger-zone-heading" className="text-lg font-medium text-destructive">
        {t("Danger Zone")}
      </h3>
      <div className="mt-4 rounded-lg border border-destructive/40 bg-destructive/5 p-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0 space-y-1 text-sm">
            <p className="font-medium text-foreground">{t("Kill switch")}</p>
            {killSwitch ? (
              <>
                <p className="text-muted-foreground">
                  {t(
                    "Calls the configured webhook to stop this agent's upstream runtime. This can cause an outage for everyone using the agent and cannot be undone from LiteLLM",
                  )}
                </p>
                <p className="font-mono break-all text-foreground">
                  {killSwitch.method ?? "POST"} {killSwitch.url}
                </p>
              </>
            ) : (
              <p className="text-muted-foreground">
                {t("Not configured. Add a kill switch webhook under Settings to enable this action")}
              </p>
            )}
          </div>
          {killSwitch && (
            <Button type="button" variant="destructive" onClick={openConfirm} disabled={isFiring} aria-busy={isFiring}>
              {isFiring && <UiLoadingSpinner className="size-4" />}
              {t("Fire Kill Switch")}
            </Button>
          )}
        </div>
        {lastResult && (
          <p className="mt-3 text-sm text-muted-foreground" role="status">
            {t("Last result: HTTP")} {lastResult.status_code}
            {lastResult.response_body ? ` ${lastResult.response_body}` : ""}
          </p>
        )}
      </div>

      <Dialog open={isConfirmOpen} onOpenChange={(open) => !open && !isFiring && setIsConfirmOpen(false)}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              {t("Fire kill switch for")} {agentName}?
            </DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <Alert variant="error">
              <CircleAlert />
              <AlertTitle>{t("This can cause an outage")}</AlertTitle>
              <AlertDescription>
                {t("LiteLLM will call")} {killSwitch?.method ?? "POST"} {killSwitch?.url}{" "}
                {t(
                  "immediately. Whatever that webhook does to the agent is outside LiteLLM's control and cannot be reverted here",
                )}
              </AlertDescription>
            </Alert>
            <div>
              <p className="mb-2 text-base font-medium text-foreground">
                {t("Type")} <span className="font-semibold text-destructive">{agentName}</span> {t("to confirm:")}
              </p>
              <InputGroup className="rounded-md">
                <InputGroupAddon>
                  <CircleAlert className="size-3.5 text-destructive" />
                </InputGroupAddon>
                <InputGroupInput
                  aria-label={t("Confirm agent name")}
                  value={confirmationInput}
                  onChange={(e) => setConfirmationInput(e.target.value)}
                  placeholder={agentName}
                  autoFocus
                />
              </InputGroup>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsConfirmOpen(false)} disabled={isFiring}>
              {t("Cancel")}
            </Button>
            <Button
              variant="destructive"
              onClick={fire}
              disabled={confirmationInput !== agentName || isFiring}
              aria-busy={isFiring}
            >
              {isFiring && <UiLoadingSpinner className="size-4" />}
              {isFiring ? t("Firing...") : t("Fire Kill Switch")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </section>
  );
};

export default AgentKillSwitchDangerZone;
