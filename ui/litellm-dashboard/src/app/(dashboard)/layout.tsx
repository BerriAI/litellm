"use client";

import React, { Suspense, useState, useRef, useEffect } from "react";
import { DashboardHeader } from "@/components/DashboardHeader";
import Navbar from "@/components/navbar";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { ThemeProvider } from "@/contexts/ThemeContext";
import { useAuth } from "@/contexts/AuthContext";
import SidebarProvider from "@/app/(dashboard)/components/SidebarProvider";
import { useRouter, useSearchParams, usePathname } from "next/navigation";
import { DebugWarningBanner } from "@/components/DebugWarningBanner";
import { LicenseExpiryBanner } from "@/components/LicenseExpiryBanner";
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
  const isGateway = mode === "ai-gateway";

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

  // Non-gateway (agent control plane) mode keeps the original full-width Navbar,
  // which carries the account menu; the redesigned sidebar + header shell is
  // scoped to the ai-gateway dashboard. Chat and the public model hub are
  // separate routes that likewise keep the old Navbar.
  if (!isGateway) {
    return (
      <div className="flex h-screen flex-col overflow-hidden bg-background">
        <Navbar accessToken={accessToken} isPublicPage={false} />
        <DebugWarningBanner accessToken={accessToken} />
        <LicenseExpiryBanner accessToken={accessToken} />
        <main className="flex min-h-0 flex-1 overflow-hidden">
          <AgentControlPlaneView />
        </main>
      </div>
    );
  }

  // Standard app shell: the viewport is fixed height and never scrolls. The
  // sidebar owns its own scroll and the content column scrolls independently,
  // so the page can't be dragged past the end of the nav.
  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <SidebarProvider
        setPage={navigateToPage}
        defaultSelectedKey={page}
        sidebarCollapsed={sidebarCollapsed}
        onToggleCollapsed={() => setSidebarCollapsed((v) => !v)}
      />
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <DashboardHeader page={page} />
        <DebugWarningBanner accessToken={accessToken} />
        <LicenseExpiryBanner accessToken={accessToken} />
        <main className="min-w-0 flex-1 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
}

function LayoutContent({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { accessToken, authLoading } = useAuth();
  const isInvitationFlow = Boolean(searchParams.get("invitation_id"));

  // Legacy invitation links point at /ui/?invitation_id=; the onboarding form now lives at its own
  // /onboarding route. Redirect once ui-config has loaded so migratedHref resolves the SERVER_ROOT_PATH base.
  useEffect(() => {
    if (!authLoading && isInvitationFlow) {
      router.replace(`${migratedHref("onboarding")}?${searchParams.toString()}`);
    }
  }, [authLoading, isInvitationFlow, router, searchParams]);

  if (authLoading || isInvitationFlow) {
    return <LoadingScreen />;
  }

  return (
    <ThemeProvider accessToken={accessToken}>
      <DashboardShell>{children}</DashboardShell>
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
