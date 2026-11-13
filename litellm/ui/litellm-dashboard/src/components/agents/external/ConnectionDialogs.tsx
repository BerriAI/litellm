import { useTranslation } from "react-i18next";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog";
import { hookUrl, Source, useExternalMutation } from "./useExternalAgents";
const secretPattern = "whsec_.+";
const modes = ["shadow", "enforce"] as const;

export function ModeControl({
  value,
  onChange,
  disabled = false,
}: {
  value: "shadow" | "enforce";
  onChange: (mode: "shadow" | "enforce") => void;
  disabled?: boolean;
}) {
  const { t } = useTranslation();

  return (
    <div className="space-y-2">
      <div
        role="group"
        aria-label={t("agents.external.governanceMode")}
        className="inline-flex rounded-lg border p-1 gap-1"
      >
        {modes.map((mode) => (
          <Button
            key={mode}
            type="button"
            size="sm"
            variant={mode === value ? "default" : "ghost"}
            disabled={disabled}
            aria-pressed={mode === value}
            onClick={() => onChange(mode)}
          >
            {mode === "shadow" ? t("agents.external.shadow") : t("agents.external.enforce")}
          </Button>
        ))}
      </div>
      <p className="text-xs text-muted-foreground">
        {t("agents.external.shadowAllowsRequestsAndRecordsWhatWouldBeDenied")}
      </p>
      <p className="text-xs text-muted-foreground">{t("agents.external.enforceBlocksRequestsThatViolatePolicy")}</p>
    </div>
  );
}

export function ConnectionDialog({
  token,
  source,
  onClose,
  onConnected,
}: {
  token: string;
  source?: Source;
  onClose: () => void;
  onConnected: (source: Source) => void;
}) {
  const { t } = useTranslation();

  const mutation = useExternalMutation(token);
  const [name, setName] = useState("");
  const [secret, setSecret] = useState("");
  const [mode, setMode] = useState<"shadow" | "enforce">("shadow");
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent showCloseButton={false}>
        <DialogHeader>
          <DialogTitle>{source ? t("agents.external.rotateSecret") : t("agents.external.connectGrokBot")}</DialogTitle>
          <DialogDescription>
            {source
              ? t("agents.external.thePreviousSecretRemainsValidDuringRotationOtherWorkersRefreshWithin20Seconds")
              : t("agents.external.connectOnceBotsAreDiscoveredAutomaticallyWithoutProxyingModelRequests")}
          </DialogDescription>
        </DialogHeader>
        <form
          className="space-y-5"
          onSubmit={async (event) => {
            event.preventDefault();
            try {
              const result = await mutation.mutateAsync({
                method: source ? "PATCH" : "POST",
                path: source ? `/sources/${source.source_id}` : "/sources",
                body: source ? { signing_secret: secret } : { display_name: name, signing_secret: secret, mode },
              });
              setSecret("");
              onConnected(result as Source);
            } catch {
              return;
            }
          }}
        >
          {!source && (
            <label className="grid gap-2">
              {t("agents.external.displayName")}
              <Input required value={name} maxLength={200} onChange={(e) => setName(e.target.value)} />
            </label>
          )}
          <label className="grid gap-2">
            {t("agents.external.grokBotSigningSecret")}
            <Input
              required
              type="password"
              autoComplete="new-password"
              placeholder={t("agents.external.whsec")}
              pattern={secretPattern}
              value={secret}
              onChange={(e) => setSecret(e.target.value)}
            />
          </label>
          {!source && <ModeControl value={mode} onChange={setMode} />}
          {mutation.error && (
            <p role="alert" className="text-destructive">
              {mutation.error.message}
            </p>
          )}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="outline" onClick={onClose}>
              {t("agents.external.cancel")}
            </Button>
            <Button type="submit" disabled={mutation.isPending}>
              {
                <>
                  {mutation.isPending && t("agents.external.saving")}
                  {!mutation.isPending && (
                    <>
                      {source && t("agents.external.saveSecret")}
                      {!source && t("agents.external.connect")}
                    </>
                  )}
                </>
              }
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
}

export function HookUrlDialog({ source, onClose }: { source: Source; onClose: () => void }) {
  const { t } = useTranslation();

  const [copied, setCopied] = useState(false);
  const [error, setError] = useState("");
  const url = hookUrl(source.source_id);
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent showCloseButton={false} className="sm:max-w-xl">
        <DialogHeader>
          <DialogTitle>{t("agents.external.grokBotHookURL")}</DialogTitle>
          <DialogDescription>{t("agents.external.oneConnectionGovernsEveryBotInYourTenant")}</DialogDescription>
        </DialogHeader>
        <label className="grid gap-2">
          {t("agents.external.hookURL")}
          <Textarea readOnly value={url} rows={3} className="resize-none font-mono text-xs" />
        </label>
        <Button
          variant="outline"
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(url);
              setCopied(true);
            } catch {
              setError(t("agents.external.copyUnavailableSelectAndCopyTheHookURLAbove"));
            }
          }}
        >
          {copied ? t("agents.external.copied") : t("agents.external.copyHookURL")}
        </Button>
        {error && <p role="alert">{error}</p>}
        <ol className="list-decimal space-y-3 pl-5 text-sm">
          <li>{t("agents.external.pasteThisURLInGrokBotGtInferenceHooks")}</li>
          <li>{t("agents.external.pasteTheSameGrokBotgeneratedSigningSecretUsedForThisConnection")}</li>
          <li>{t("agents.external.runTestConnectionInGrokBotBotsAppearAfterTheirFirstRealEvent")}</li>
        </ol>
        <Button onClick={onClose}>{t("agents.external.done")}</Button>
      </DialogContent>
    </Dialog>
  );
}
