import { useTranslation } from "react-i18next";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Connection, hookUrl, useExternalMutation, useExternalQuery } from "./useExternalAgents";

const modes = ["shadow", "enforce"] as const;
const secretPattern = "whsec_.+";

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
    <div
      role="group"
      aria-label={t("agents.external.governanceMode")}
      className="inline-flex rounded-lg border p-1 gap-1"
    >
      {modes.map((mode) => (
        <Button
          key={mode}
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
  );
}

function CopyField({ label, value, password = false }: { label: string; value: string; password?: boolean }) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState("");
  return (
    <div className="space-y-2">
      <label className="grid gap-2 text-sm">
        {label}
        <Input readOnly type={password ? "password" : "text"} value={value} />
      </label>
      <Button
        variant="outline"
        size="sm"
        aria-label={t("agents.external.copyField", { label })}
        onClick={async () => {
          try {
            await navigator.clipboard.writeText(value);
            setCopied(true);
          } catch {
            setError(t("agents.external.copyUnavailable"));
          }
        }}
      >
        {copied ? t("agents.external.copied") : t("agents.external.copy")}
      </Button>
      {error && <p role="alert">{error}</p>}
    </div>
  );
}

function OptionalSigningSecret({ token, sourceId }: { token: string; sourceId: string }) {
  const { t } = useTranslation();
  const [value, setValue] = useState("");
  const [saved, setSaved] = useState(false);
  const mutation = useExternalMutation(token);
  const title = t("agents.external.optionalSigningSecret");
  return (
    <details className="rounded-lg border p-4">
      <summary className="cursor-pointer font-medium">{title}</summary>
      <form
        aria-label={title}
        className="mt-4 space-y-3"
        onSubmit={async (event) => {
          event.preventDefault();
          try {
            await mutation.mutateAsync({
              method: "PATCH",
              path: `/sources/${sourceId}`,
              body: { signing_secret: value },
            });
            setValue("");
            setSaved(true);
          } catch {
            setSaved(false);
          }
        }}
      >
        <label className="grid gap-2 text-sm">
          {t("agents.external.grokBotSigningSecret")}
          <Input
            required
            type="password"
            autoComplete="new-password"
            value={value}
            pattern={secretPattern}
            onChange={(event) => setValue(event.target.value)}
          />
        </label>
        <Button type="submit" disabled={mutation.isPending}>
          {t("agents.external.save")}
        </Button>
        {saved && <p role="status">{t("agents.external.saved")}</p>}
        {mutation.error && (
          <p role="alert" className="text-destructive">
            {mutation.error.message}
          </p>
        )}
      </form>
    </details>
  );
}

export function SetupCard({ token, sourceId, status }: { token: string; sourceId: string; status: string }) {
  const { t } = useTranslation();
  const connection = useExternalQuery<Connection>(token, `/sources/${sourceId}/connection`);
  return (
    <div className="rounded-xl border bg-card p-6 space-y-5" aria-label={t("agents.external.setupGrokBot")}>
      <h2 className="text-lg font-semibold">{t("agents.external.setupGrokBot")}</h2>
      {connection.isPending && <p role="status">{t("agents.external.loadingConnections")}</p>}
      {connection.error && <p role="alert">{connection.error.message}</p>}
      {connection.data && (
        <ol className="list-decimal space-y-5 pl-5">
          <li>
            <CopyField label={t("agents.external.endpointUrl")} value={hookUrl(sourceId)} />
          </li>
          <li className="space-y-3">
            <CopyField label={t("agents.external.customHeaderName")} value="Authorization" />
            <CopyField
              label={t("agents.external.customHeaderValue")}
              value={`Bearer ${connection.data.ingest_key}`}
              password
            />
          </li>
          <li className="text-sm">{t("agents.external.setupInstructions")}</li>
        </ol>
      )}
      <p role="status" className="font-medium">
        {status}
      </p>
      <p className="text-xs text-muted-foreground">
        {t("agents.external.shadowAllowsRequestsAndRecordsWhatWouldBeDenied")}
      </p>
      <OptionalSigningSecret token={token} sourceId={sourceId} />
    </div>
  );
}
