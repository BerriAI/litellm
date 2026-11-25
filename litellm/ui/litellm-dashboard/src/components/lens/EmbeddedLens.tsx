"use client";

import { useCallback, useEffect, useEffectEvent, useLayoutEffect, useRef, useState } from "react";
import { useTheme } from "next-themes";
import { useTranslation } from "react-i18next";
import { getProxyBaseUrl } from "@/components/networking";
import { Button, buttonVariants } from "@/components/ui/button";
import {
  LENS_UI_AUTH,
  lensWorkspaceAvailability,
  type LensCredentials,
  type LensWorkspaceAvailability,
} from "@/lib/http/lensWorkspace";
import { serverRootPath } from "@/lib/serverRootPath";
import { resolveApiBase } from "@/lib/http/resolveApiBase";
import { LensIntroduction } from "./LensIntroduction";
import { LENS_FRAME_CHANNEL, lensFrameMessage, useSetLensNavigation } from "./GatewayLensNavigation";

type Workspace = { state: "checking" } | LensWorkspaceAvailability;

const gatewayUrl = (path: string): string =>
  `${resolveApiBase({ explicitBase: getProxyBaseUrl(), serverRootPath })}${path}`;

function LensFrame({ source, accessToken, headerName }: LensCredentials & { source: string }) {
  const { t } = useTranslation();
  const { resolvedTheme } = useTheme();
  const iframe = useRef<HTMLIFrameElement>(null);
  const setNavigation = useSetLensNavigation();
  const origin = new URL(source, window.location.href).origin;
  const theme = resolvedTheme === "dark" ? "dark" : "light";
  const sendTheme = useCallback(() => {
    iframe.current?.contentWindow?.postMessage(
      { channel: LENS_FRAME_CHANNEL, command: { type: "theme", theme } },
      origin,
    );
  }, [origin, theme]);
  const receive = useEffectEvent((event: MessageEvent<unknown>) => {
    if (origin !== window.location.origin) return;
    const child = iframe.current?.contentWindow;
    if (!child || event.source !== child || event.origin !== origin) return;
    const parsed = lensFrameMessage.safeParse(event.data);
    if (!parsed.success) return;
    if (parsed.data.type === "auth-ready") {
      child.postMessage({ channel: LENS_FRAME_CHANNEL, command: { type: "auth", accessToken, headerName } }, origin);
    } else if (parsed.data.type === "ready") {
      setNavigation(null);
      sendTheme();
    } else {
      setNavigation({
        model: parsed.data.navigation,
        select: (command) => child.postMessage({ channel: LENS_FRAME_CHANNEL, command }, origin),
      });
    }
  });
  useLayoutEffect(() => {
    const listener = (event: MessageEvent<unknown>) => receive(event);
    window.addEventListener("message", listener);
    return () => {
      window.removeEventListener("message", listener);
      setNavigation(null);
    };
  }, [setNavigation]);
  useEffect(() => {
    sendTheme();
  }, [sendTheme]);

  return (
    <iframe
      ref={iframe}
      title={t("lensWorkspace.frameTitle")}
      src={source}
      data-lens-auth={LENS_UI_AUTH}
      className="h-full w-full flex-1 border-0"
    />
  );
}

export function EmbeddedLens({ accessToken, headerName }: LensCredentials) {
  return (
    <LensConnection key={JSON.stringify([accessToken, headerName])} accessToken={accessToken} headerName={headerName} />
  );
}

function LensConnection({ accessToken, headerName }: LensCredentials) {
  const { t } = useTranslation();
  const [workspace, setWorkspace] = useState<Workspace>({ state: "checking" });
  const [attempt, setAttempt] = useState(0);
  const [source] = useState(() => {
    const url = new URL(gatewayUrl("/lens/ui/"), window.location.href);
    url.search = window.location.search;
    return url.href;
  });
  const retry = () => {
    setWorkspace({ state: "checking" });
    setAttempt((current) => current + 1);
  };

  useEffect(() => {
    const controller = new AbortController();
    void lensWorkspaceAvailability(gatewayUrl("/lens/ui/"), { accessToken, headerName }, controller.signal).then(
      (availability) => {
        if (controller.signal.aborted) return;
        setWorkspace(availability);
      },
    );
    return () => controller.abort();
  }, [attempt, accessToken, headerName]);

  if (workspace.state === "checking")
    return (
      <p role="status" className="p-8 text-sm text-muted-foreground">
        {t("lensWorkspace.opening")}
      </p>
    );
  if (workspace.state === "served")
    return <LensFrame source={source} accessToken={accessToken} headerName={headerName} />;
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
