"use client";

import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { getProxyBaseUrl } from "@/components/networking";
import { Button, buttonVariants } from "@/components/ui/button";
import { lensWorkspaceAvailability, type LensWorkspaceAvailability } from "@/lib/http/lensWorkspace";
import { serverRootPath } from "@/lib/serverRootPath";
import { LensIntroduction } from "./LensIntroduction";

type Workspace = { state: "checking" } | LensWorkspaceAvailability;

const gatewayUrl = (path: string): string =>
  `${getProxyBaseUrl().replace(/\/$/, "")}${serverRootPath.replace(/\/$/, "")}${path}`;

export function EmbeddedLens() {
  const { t } = useTranslation();
  const [workspace, setWorkspace] = useState<Workspace>({ state: "checking" });
  const [attempt, setAttempt] = useState(0);
  const [source] = useState(() => gatewayUrl("/lens/ui/") + window.location.search);
  const retry = () => {
    setWorkspace({ state: "checking" });
    setAttempt((current) => current + 1);
  };

  useEffect(() => {
    const controller = new AbortController();
    void lensWorkspaceAvailability(gatewayUrl("/lens/ui/"), controller.signal).then((availability) => {
      if (!controller.signal.aborted) setWorkspace(availability);
    });
    return () => controller.abort();
  }, [attempt]);

  if (workspace.state === "checking")
    return (
      <p role="status" className="p-8 text-sm text-muted-foreground">
        {t("lensWorkspace.opening")}
      </p>
    );
  if (workspace.state === "served")
    return <iframe title={t("lensWorkspace.frameTitle")} src={source} className="h-full w-full flex-1 border-0" />;
  if (workspace.state === "unconfigured") return <LensIntroduction onRetry={retry} />;
  return (
    <div className="flex min-h-96 flex-1 items-center justify-center px-6 py-16 sm:px-10">
      <section className="w-full max-w-lg" aria-labelledby="lens-connection-title">
        <h1 id="lens-connection-title" className="text-2xl font-semibold tracking-tight sm:text-3xl">
          {t(`lensWorkspace.${workspace.state}.title`)}
        </h1>
        <p className="mt-4 text-sm leading-6 text-muted-foreground">{t(`lensWorkspace.${workspace.state}.body`)}</p>
        {workspace.state === "unsupported" && !workspace.browserUrl && (
          <p className="mt-3 text-sm leading-6 text-muted-foreground">{t("lensWorkspace.unsupported.noBrowserUrl")}</p>
        )}
        <div className="mt-6 flex flex-wrap items-center gap-3">
          {workspace.state === "unsupported" && workspace.browserUrl && (
            <a className={buttonVariants()} href={gatewayUrl("/lens/login")} target="_blank" rel="noreferrer">
              {t("lensWorkspace.unsupported.open")}
            </a>
          )}
          <Button variant="outline" onClick={retry}>
            {t("lensWorkspace.retry")}
          </Button>
          <a
            className={buttonVariants({ variant: "link" })}
            href="https://docs.litellm.ai/docs/proxy/lens"
            target="_blank"
            rel="noreferrer"
          >
            {t("lensWorkspace.docs")}
          </a>
        </div>
      </section>
    </div>
  );
}
