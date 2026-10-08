"use client";

import { Suspense, useEffect, useMemo, useSyncExternalStore } from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import Navbar from "@/components/navbar";
import MoyaiConnected from "@/components/moyai/MoyaiConnected";
import MoyaiLanding from "@/components/moyai/MoyaiLanding";
import { startMoyaiQuickConnect } from "@/components/networking";
import { PluginModeProvider } from "@/contexts/PluginModeContext";
import { ThemeProvider } from "@/contexts/ThemeContext";
import { isProxyAdminRole } from "@/utils/roles";
import { uiHref } from "@/utils/uiHref";

function MoyaiPageContent() {
  const { accessToken, userRole } = useAuthorized();
  const { data: uiSettings, isLoading, refetch } = useUISettings();

  const parsed = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false,
  );
  const connectedParams = useMemo(() => {
    if (!parsed) {
      return null;
    }
    const params = new URLSearchParams(window.location.search);
    if (params.get("moyai_connected") !== "1") {
      return null;
    }
    const models = Number(params.get("models"));
    return {
      keyAlias: params.get("key_alias"),
      models: Number.isFinite(models) && params.get("models") !== null ? models : null,
    };
  }, [parsed]);

  useEffect(() => {
    if (connectedParams) {
      window.history.replaceState(null, "", window.location.pathname);
      refetch();
    }
  }, [connectedParams, refetch]);

  const moyaiUrl = (uiSettings?.values?.moyai_url as string | undefined) ?? null;

  const settingsReady = parsed && !isLoading;
  const shouldOpenMoyai = settingsReady && !connectedParams && moyaiUrl;
  useEffect(() => {
    if (shouldOpenMoyai) {
      window.location.replace(moyaiUrl as string);
    }
  }, [shouldOpenMoyai, moyaiUrl]);

  const onQuickConnect = async (url: string) => {
    const response = await startMoyaiQuickConnect(accessToken ?? "", url, window.location.origin + uiHref("moyai"));
    window.location.assign(response.connect_url);
  };

  let content: React.ReactNode = null;
  if (parsed && !isLoading) {
    if (connectedParams) {
      content = (
        <MoyaiConnected moyaiUrl={moyaiUrl} keyAlias={connectedParams.keyAlias} models={connectedParams.models} />
      );
    } else if (moyaiUrl) {
      content = (
        <div className="flex min-h-[calc(100vh-3.5rem)] flex-col items-center justify-center gap-3 bg-[#04060c] px-6 text-center text-[#e3e3e3]">
          <p className="m-0 text-lg font-medium text-white">Opening Moyai</p>
          <a href={moyaiUrl} className="text-sm text-[#8b9bff] underline underline-offset-4">
            Continue to {moyaiUrl}
          </a>
        </div>
      );
    } else {
      content = <MoyaiLanding canQuickConnect={isProxyAdminRole(userRole ?? "")} onQuickConnect={onQuickConnect} />;
    }
  }

  return (
    <PluginModeProvider accessToken={accessToken}>
      <ThemeProvider accessToken={accessToken}>
        <div className="flex h-screen flex-col">
          <Navbar accessToken={accessToken} isPublicPage={false} />
          <main className="min-h-0 flex-1 overflow-y-auto">{content}</main>
        </div>
      </ThemeProvider>
    </PluginModeProvider>
  );
}

export default function MoyaiPage() {
  return (
    <Suspense>
      <MoyaiPageContent />
    </Suspense>
  );
}
