import { useState } from "react";
import { formatDistanceToNow } from "date-fns";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import {
  AlertDialog,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useUrlTab } from "@/hooks/useUrlTab";
import { Connection, useCloudMutation, useCloudSources } from "./hooks";
import Setup, { ConnectionStatus } from "./Setup";
import Bots from "./Bots";
import Spend from "./Spend";

export default function ExternalCloudAgents({ accessToken }: { accessToken: string }) {
  const { t } = useTranslation();
  const sources = useCloudSources(accessToken);
  const source = sources.data?.[0];
  const sourceId = source?.source_id;
  const [connection, setConnection] = useState<Connection | null>(null);
  const [showSetup, setShowSetup] = useState(false);
  const [confirmRemove, setConfirmRemove] = useState(false);
  const [tab, setTab] = useUrlTab(["bots", "spend"], "bots", "cloud_tab");
  const create = useCloudMutation<Connection>(accessToken, "POST", "/sources");
  const reveal = useCloudMutation<Connection>(accessToken, "GET", `/sources/${sourceId}/connection`);
  const remove = useCloudMutation(accessToken, "DELETE", `/sources/${sourceId}`);
  const error = [sources.error, create.error, reveal.error, remove.error].find(Boolean);
  if (sources.isLoading)
    return (
      <p role="status" className="p-8">
        {t("agents.external.loading")}
      </p>
    );
  return (
    <section className="min-w-0 space-y-6 p-6 md:p-8">
      <h1 className="text-2xl font-bold">{t("agents.external.title")}</h1>
      {error && (
        <div role="alert" className="flex items-center gap-3 text-sm text-destructive">
          {error.message}
          <Button variant="outline" onClick={() => sources.refetch()}>
            {t("agents.external.retry")}
          </Button>
        </div>
      )}
      {!source && !connection ? (
        <div className="flex min-h-72 flex-col items-center justify-center gap-5 rounded-xl border border-dashed bg-card p-8 text-center">
          <p className="max-w-lg text-muted-foreground">{t("agents.external.empty")}</p>
          <Button
            disabled={create.isPending}
            onClick={() => create.mutate(undefined, { onSuccess: (created) => setConnection(created) })}
          >
            {t("agents.external.add")}
          </Button>
        </div>
      ) : (
        <>
          {source?.connected && (
            <header className="flex flex-wrap items-center gap-4 rounded-lg border bg-card px-5 py-4">
              <h2 className="font-semibold">{source.display_name}</h2>
              <ConnectionStatus source={source} />
              <span className="text-sm text-muted-foreground">
                {source.last_event_at &&
                  t("agents.external.lastRequest", {
                    time: formatDistanceToNow(new Date(source.last_event_at), { addSuffix: true }),
                  })}
              </span>
              <div className="ml-auto flex gap-2">
                <Button
                  variant="outline"
                  disabled={reveal.isPending}
                  onClick={() => {
                    if (showSetup) {
                      setShowSetup(false);
                      setConnection(null);
                    } else {
                      reveal.mutate(undefined, {
                        onSuccess: (value) => {
                          setConnection(value);
                          setShowSetup(true);
                        },
                      });
                    }
                  }}
                >
                  {showSetup ? t("agents.external.hideSetup") : t("agents.external.showSetup")}
                </Button>
                <Button variant="outline" onClick={() => setConfirmRemove(true)}>
                  {t("agents.external.remove")}
                </Button>
              </div>
            </header>
          )}
          {(!source?.connected || showSetup) &&
            (connection ? (
              <Setup connection={connection} source={source} />
            ) : (
              <div className="space-y-4">
                <ConnectionStatus source={source} />
                <Button
                  disabled={reveal.isPending}
                  onClick={() => reveal.mutate(undefined, { onSuccess: setConnection })}
                >
                  {t("agents.external.showSetup")}
                </Button>
              </div>
            ))}
          {source?.connected && (
            <>
              <div className="flex gap-2 border-b pb-3" role="tablist" aria-label={t("agents.external.title")}>
                <Button
                  role="tab"
                  aria-selected={tab === "bots"}
                  variant={tab === "bots" ? "default" : "ghost"}
                  onClick={() => setTab("bots")}
                >
                  {t("agents.external.bots")}
                </Button>
                <Button
                  role="tab"
                  aria-selected={tab === "spend"}
                  variant={tab === "spend" ? "default" : "ghost"}
                  onClick={() => setTab("spend")}
                >
                  {t("agents.external.spend")}
                </Button>
              </div>
              {tab === "bots" ? (
                <Bots accessToken={accessToken} sourceId={source.source_id} />
              ) : (
                <Spend accessToken={accessToken} source={source} />
              )}
            </>
          )}
        </>
      )}
      <AlertDialog open={confirmRemove} onOpenChange={setConfirmRemove}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("agents.external.removeTitle")}</AlertDialogTitle>
            <AlertDialogDescription>{t("agents.external.removeDescription")}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <Button variant="outline" onClick={() => setConfirmRemove(false)}>
              {t("agents.external.cancel")}
            </Button>
            <Button
              variant="destructive"
              disabled={remove.isPending}
              onClick={() =>
                remove.mutate(undefined, {
                  onSuccess: () => {
                    setConnection(null);
                    setShowSetup(false);
                    setConfirmRemove(false);
                  },
                })
              }
            >
              {t("agents.external.remove")}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </section>
  );
}
