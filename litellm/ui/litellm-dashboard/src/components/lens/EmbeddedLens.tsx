"use client";

import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { getProxyBaseUrl } from "@/components/networking";
import { buttonVariants } from "@/components/ui/button";
import { lensWorkspaceAvailability, type LensWorkspaceAvailability } from "@/lib/http/lensWorkspace";
import { serverRootPath } from "@/lib/serverRootPath";

type Workspace = "checking" | LensWorkspaceAvailability;

const gatewayUrl = (path: string): string =>
  `${getProxyBaseUrl().replace(/\/$/, "")}${serverRootPath.replace(/\/$/, "")}${path}`;

/**
 * The Lens page is the UI of the Lens this gateway is connected to, framed as that Lens serves it. The gateway
 * forwards the UI and the APIs its pages call under /lens/, signed in with the dashboard login the browser holds.
 */
export function EmbeddedLens() {
  const { t } = useTranslation();
  const [workspace, setWorkspace] = useState<Workspace>("checking");
  const [source] = useState(() => gatewayUrl("/lens/ui/") + window.location.search);

  useEffect(() => {
    const controller = new AbortController();
    void lensWorkspaceAvailability(gatewayUrl("/lens/ui/"), controller.signal).then((availability) => {
      if (!controller.signal.aborted) setWorkspace(availability);
    });
    return () => controller.abort();
  }, []);

  if (workspace === "checking")
    return (
      <p role="status" className="p-8 text-sm text-muted-foreground">
        {t("lensWorkspace.opening")}
      </p>
    );
  if (workspace === "served")
    return <iframe title={t("lensWorkspace.frameTitle")} src={source} className="h-full w-full flex-1 border-0" />;
  return (
    <div className="flex max-w-xl flex-col items-start gap-3 p-8">
      <h1 className="text-lg font-semibold">{t(`lensWorkspace.${workspace}.title`)}</h1>
      <p className="text-sm text-muted-foreground">{t(`lensWorkspace.${workspace}.body`)}</p>
      {workspace === "unsupported" ? (
        <a className={buttonVariants()} href={gatewayUrl("/lens/login")} target="_blank" rel="noreferrer">
          {t("lensWorkspace.unsupported.open")}
        </a>
      ) : null}
    </div>
  );
}
