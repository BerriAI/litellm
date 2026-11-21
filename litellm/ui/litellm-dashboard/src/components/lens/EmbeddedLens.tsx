"use client";

import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { getGlobalLitellmHeaderName, getProxyBaseUrl } from "@/components/networking";
import { buttonVariants } from "@/components/ui/button";
import { lensWorkspaceAvailability, type LensWorkspaceAvailability } from "@/lib/http/lensWorkspace";
import { getAuthToken } from "@/lib/http/runtime";
import { serverRootPath } from "@/lib/serverRootPath";

const EMBED_PROTOCOL = 1;

type Workspace = "checking" | LensWorkspaceAvailability;

type EmbeddedLensProps = {
  readonly accessToken: string;
  readonly userRole: string;
  readonly readOnly: boolean;
};

const gatewayUrl = (path: string): string =>
  `${getProxyBaseUrl().replace(/\/$/, "")}${serverRootPath.replace(/\/$/, "")}${path}`;

type FrameMessage = { type?: unknown; protocol?: unknown; search?: unknown };

const isFrameMessage = (data: unknown, type: string): data is FrameMessage => {
  if (typeof data !== "object" || data === null) return false;
  const message: FrameMessage = data;
  return message.type === type && message.protocol === EMBED_PROTOCOL;
};

export function EmbeddedLens({ accessToken, userRole, readOnly }: EmbeddedLensProps) {
  const { t } = useTranslation();
  const frame = useRef<HTMLIFrameElement>(null);
  const [workspace, setWorkspace] = useState<Workspace>("checking");
  const [source] = useState(() => gatewayUrl("/lens/ui/") + window.location.search);

  useEffect(() => {
    const controller = new AbortController();
    void lensWorkspaceAvailability(gatewayUrl("/lens/ui/"), controller.signal).then((availability) => {
      if (!controller.signal.aborted) setWorkspace(availability);
    });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const origin = new URL(source, window.location.href).origin;
    const receive = (event: MessageEvent) => {
      const target = frame.current?.contentWindow;
      if (!target || event.source !== target || event.origin !== origin) return;
      if (isFrameMessage(event.data, "lens:ready")) {
        const host = {
          type: "lens:host",
          protocol: EMBED_PROTOCOL,
          apiBaseUrl: getProxyBaseUrl(),
          authToken: getAuthToken() ?? accessToken,
          authHeaderName: getGlobalLitellmHeaderName(),
          serverRootPath,
          userRole,
          readOnly,
        };
        target.postMessage(host, origin);
        return;
      }
      if (isFrameMessage(event.data, "lens:location") && typeof event.data.search === "string") {
        const search = event.data.search ? `?${event.data.search}` : "";
        window.history.replaceState(window.history.state, "", window.location.pathname + search);
      }
    };
    window.addEventListener("message", receive);
    return () => window.removeEventListener("message", receive);
  }, [accessToken, readOnly, source, userRole]);

  if (workspace === "checking")
    return (
      <p role="status" className="p-8 text-sm text-muted-foreground">
        {t("lensWorkspace.opening")}
      </p>
    );
  if (workspace === "served")
    return (
      <iframe
        ref={frame}
        title={t("lensWorkspace.frameTitle")}
        src={source}
        className="h-full w-full flex-1 border-0"
      />
    );
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
