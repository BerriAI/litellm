"use client";

import React, { Suspense, useState, useRef, useEffect } from "react";
import Navbar from "@/components/navbar";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { ThemeProvider } from "@/contexts/ThemeContext";
import { useAuth } from "@/contexts/AuthContext";
import SidebarProvider from "@/app/(dashboard)/components/SidebarProvider";
import { useRouter, useSearchParams, usePathname } from "next/navigation";
import { DebugWarningBanner } from "@/components/DebugWarningBanner";
import { MIGRATED_PAGES, migratedHref, legacyPageHref, legacyKeyForPathname } from "@/utils/migratedPages";
import { PluginModeProvider, usePluginMode } from "@/contexts/PluginModeContext";
import { createApiClient } from "@/lib/http/client";
import { getProxyBaseUrl } from "@/components/networking";

const pluginApiClient = createApiClient({ getBaseUrl: () => getProxyBaseUrl() ?? "" });

// Wrapper so PluginModeProvider receives the live accessToken from auth context,
// which means plugin data refreshes on login/logout without stale cookie reads.
function PluginModeProviderWithAuth({ children }: { children: React.ReactNode }) {
  const { accessToken } = useAuth();
  return <PluginModeProvider accessToken={accessToken}>{children}</PluginModeProvider>;
}

export function AgentControlPlaneView() {
  const { activePlugin } = usePluginMode();
  const activePluginName = activePlugin?.name;
  const agentPlatformUrl = activePlugin?.url ?? "";
  const { accessToken } = useAuth();
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [auth, setAuth] = useState<{ plugin: string; claim: string } | null>(null);

  // Fetch a short-lived identity claim scoped to the *active* plugin. The claim
  // is encrypted under that plugin's own per-plugin key, so it must be requested
  // per plugin and re-fetched when the user switches plugins.
  useEffect(() => {
    if (!accessToken || !activePluginName) return;
    let cancelled = false;
    pluginApiClient
      .get("/api/plugins/auth-token", { accessToken, query: { plugin_name: activePluginName } })
      .then((data: { session_claim?: string }) => {
        if (!cancelled && data?.session_claim) setAuth({ plugin: activePluginName, claim: data.session_claim });
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [accessToken, activePluginName]);

  // Deliver the claim to the iframe via postMessage, but only while it was issued
  // for the plugin currently mounted — never replay one plugin's claim to another.
  // targetOrigin is the configured plugin URL — no other origin receives it.
  useEffect(() => {
    const iframe = iframeRef.current;
    if (!iframe || !auth || auth.plugin !== activePluginName || !agentPlatformUrl) return;
    const send = () => {
      iframe.contentWindow?.postMessage({ type: "litellm-auth", session_claim: auth.claim }, agentPlatformUrl);
    };
    // Cover both orderings: the iframe may have already fired `load` before the
    // claim arrived (send now), or it may load/reload later (send on the event).
    send();
    iframe.addEventListener("load", send);
    return () => iframe.removeEventListener("load", send);
  }, [auth, activePluginName, agentPlatformUrl]);

  if (!agentPlatformUrl) {
    return (
      <div className="flex flex-1 items-center justify-center text-gray-500">
        <div className="text-center">
          <p className="text-lg font-medium mb-2">Plugin</p>
          <p className="text-sm">Configure the plugin URL in settings</p>
        </div>
      </div>
    );
  }

  // Embed the plugin at its root; the plugin renders its own full UI (incl. nav) inside.
  return (
    <iframe
      ref={iframeRef}
      src={`${agentPlatformUrl.replace(/\/$/, "")}/`}
      style={{
        width: "100%",
        height: "100%",
        border: "none",
        flex: 1,
        minHeight: "calc(100vh - 56px)",
      }}
      title={activePlugin?.display_name ?? "Plugin"}
      allow="clipboard-write"
    />
  );
}

function DashboardShell({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const pathname = usePathname();
  const { accessToken } = useAuth();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const { mode } = usePluginMode();

  const page = legacyKeyForPathname(pathname) || searchParams.get("page") || "api-keys";

  const navigateToPage = (newPage: string) => {
    const migratedRoute = MIGRATED_PAGES[newPage];
    const href = migratedRoute ? migratedHref(migratedRoute) : legacyPageHref(newPage);
    // Clicking the already-active sidebar item (e.g. Virtual Keys while
    // ?virtual_key= is open) is often a Next no-op; clear leftover deep-link
    // query params so refresh does not reopen the detail view.
    if (typeof window !== "undefined") {
      const targetPath = href.split("?")[0];
      if (window.location.pathname === targetPath && window.location.search) {
        window.history.replaceState(null, "", targetPath);
        window.dispatchEvent(new PopStateEvent("popstate"));
      }
    }
    router.push(href);
  };

  return (
    <div className="flex flex-col min-h-screen">
      <Navbar
        accessToken={accessToken}
        isPublicPage={false}
        sidebarCollapsed={sidebarCollapsed}
        onToggleSidebar={() => setSidebarCollapsed((v) => !v)}
      />
      <DebugWarningBanner accessToken={accessToken} />
      <div className="flex flex-1">
        {mode !== "ai-gateway" ? (
          <div className="flex-1 flex">
            <AgentControlPlaneView />
          </div>
        ) : (
          <>
            <div className="mt-2">
              <SidebarProvider setPage={navigateToPage} defaultSelectedKey={page} sidebarCollapsed={sidebarCollapsed} />
            </div>
            <main className="flex-1 min-w-0">{children}</main>
          </>
        )}
      </div>
    </div>
  );
}

function LayoutContent({ children }: { children: React.ReactNode }) {
  const searchParams = useSearchParams();
  const { accessToken, authLoading } = useAuth();
  const isInvitationFlow = Boolean(searchParams.get("invitation_id"));

  if (authLoading) {
    return <LoadingScreen />;
  }

  return (
    <ThemeProvider accessToken={accessToken}>
      {isInvitationFlow ? children : <DashboardShell>{children}</DashboardShell>}
    </ThemeProvider>
  );
}

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <Suspense fallback={<LoadingScreen />}>
      <PluginModeProviderWithAuth>
        <LayoutContent>{children}</LayoutContent>
      </PluginModeProviderWithAuth>
    </Suspense>
  );
}
