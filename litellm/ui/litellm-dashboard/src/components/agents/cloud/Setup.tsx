import { useState } from "react";
import { useTranslation } from "react-i18next";
import { getProxyBaseUrl } from "@/components/networking";
import { Button } from "@/components/ui/button";
import { Connection, Source } from "./hooks";

export function ConnectionStatus({ source }: { source?: Source }) {
  const { t } = useTranslation();
  return (
    <p role="status" className="flex items-center gap-2 text-sm">
      <span
        aria-hidden
        className={`h-2 w-2 shrink-0 rounded-full ${source?.connected ? "bg-emerald-500" : "animate-pulse bg-amber-400"}`}
      />
      {source?.connected ? t("agents.external.connected", { count: source.bot_count }) : t("agents.external.waiting")}
    </p>
  );
}

function CopyValue({ label, value }: { label: string; value: string }) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      setError(false);
    } catch {
      setError(true);
    }
  };
  return (
    <div className="space-y-2">
      <p className="text-sm font-medium">{label}</p>
      <div className="flex items-start gap-3">
        <code className="min-w-0 flex-1 break-all rounded-md border bg-muted/40 px-3 py-2.5 text-xs">{value}</code>
        <Button variant="outline" onClick={copy}>
          {copied ? t("agents.external.copied") : t("agents.external.copy")}
        </Button>
      </div>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {t("agents.external.copyFailed")}
        </p>
      )}
    </div>
  );
}

export default function Setup({ connection, source }: { connection: Connection; source?: Source }) {
  const { t } = useTranslation();
  const url = new URL(connection.hook_path, getProxyBaseUrl() || window.location.origin).href;
  return (
    <section
      className="mx-auto max-w-3xl rounded-xl border bg-card p-8 shadow-sm"
      aria-label={t("agents.external.setup")}
    >
      <h2 className="mb-6 text-xl font-semibold">{t("agents.external.setup")}</h2>
      <ol className="space-y-6">
        <li className="flex gap-4">
          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-muted font-semibold">
            1
          </span>
          <div className="min-w-0 flex-1">
            <CopyValue label={t("agents.external.endpoint")} value={url} />
          </div>
        </li>
        <li className="flex gap-4">
          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-muted font-semibold">
            2
          </span>
          <div className="min-w-0 flex-1 space-y-4">
            <CopyValue label={t("agents.external.headerName")} value="Authorization" />
            <CopyValue label={t("agents.external.headerValue")} value={`Bearer ${connection.ingest_key}`} />
          </div>
        </li>
        <li className="flex gap-4">
          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-muted font-semibold">
            3
          </span>
          <p className="pt-0.5 text-sm leading-relaxed">{t("agents.external.instructions")}</p>
        </li>
      </ol>
      <div className="mt-8 border-t pt-5">
        <ConnectionStatus source={source} />
      </div>
    </section>
  );
}
