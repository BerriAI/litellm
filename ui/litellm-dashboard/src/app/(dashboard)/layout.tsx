"use client";

import React, { Suspense, useState, useRef, useEffect } from "react";
import { useTranslation } from "react-i18next";
import { DashboardHeader } from "@/components/DashboardHeader";
import Navbar from "@/components/navbar";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { ThemeProvider } from "@/contexts/ThemeContext";
import { useAuth } from "@/contexts/AuthContext";
import { useCurrentUser } from "@/app/(dashboard)/hooks/currentUser/useCurrentUser";
import { redirectToLogin } from "@/app/(dashboard)/hooks/redirectToLogin";
import { clearTokenCookies } from "@/utils/cookieUtils";
import { decodeToken } from "@/utils/jwtUtils";
import AppSidebar from "@/app/(dashboard)/components/AppSidebar";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  createDeploymentHealthHandle,
  DeploymentHealthBanner,
  DeploymentHealthIcon,
  DeploymentHealthLink,
} from "@/components/DeploymentHealth/DeploymentHealth";
import { DeploymentHealthPanel } from "@/components/DeploymentHealth/DeploymentHealthPanel";
import { UserBanner } from "@/components/UserBanner";
import { LiteAdminFrame } from "@/components/liteadmin/LiteAdmin";
import { routeSegmentForPathname, uiHref } from "@/utils/uiHref";
import { PluginModeProvider, usePluginMode } from "@/contexts/PluginModeContext";
import { createApiClient } from "@/lib/http/client";
import { getProxyBaseUrl } from "@/components/networking";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { MobileSidebarTrigger, useCloseMobileSidebarOn } from "@/components/shared/MobileSidebar";
import { sidebarProviderClassName, sidebarWidthStyle } from "@/components/shared/sidebarStyles";
import { cn } from "@/lib/cva.config";
import { CommandPaletteProvider } from "@/components/CommandPalette/CommandPaletteProvider";
import { recordUiEvent } from "@/lib/telemetry/uiEvents";
import TelemetryEnvBanner from "@/app/(dashboard)/telemetry/_components/TelemetryEnvBanner";

const pluginApiClient = createApiClient({ getBaseUrl: () => getProxyBaseUrl() ?? "" });

// Wrapper so PluginModeProvider receives the live accessToken from auth context,
// which means plugin data refreshes on login/logout without stale cookie reads.
function PluginModeProviderWithAuth({ children }: { children: React.ReactNode }) {
  const { accessToken } = useAuth();
  return <PluginModeProvider accessToken={accessToken}>{children}</PluginModeProvider>;
}

export function AgentControlPlaneView() {
  const { t } = useTranslation();
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
      <div className="flex flex-1 items-center justify-center text-muted-foreground">
        <div className="text-center">
          <p className="text-lg font-medium mb-2">{t("shell.agentControlPlane.plugin")}</p>
          <p className="text-sm">{t("shell.agentControlPlane.configure")}</p>
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
      title={activePlugin?.display_name ?? t("shell.agentControlPlane.plugin")}
      allow="clipboard-write"
    />
  );
}

const FULL_BLEED_SEGMENTS = new Set(["logs"]);

function CloseMobileSidebarOnNavigation({ navigationKey }: { navigationKey: string }) {
  useCloseMobileSidebarOn(navigationKey);
  return null;
}
const CUSTOMER_PORTAL_SEGMENTS = new Set(["customers", "chargebacks"]);

function DashboardShell({ children }: { children: React.ReactNode }) {
  const { t } = useTranslation();
  const { accessToken } = useAuth();
  const { mode, setMode } = usePluginMode();
  const pathname = usePathname();
  const routeSegment = routeSegmentForPathname(pathname);
  const isPortalRoute = CUSTOMER_PORTAL_SEGMENTS.has(routeSegment);
  useEffect(() => {
    if (isPortalRoute && mode !== "customer-portal") setMode("customer-portal");
    if (!isPortalRoute && mode === "customer-portal") setMode("ai-gateway");
  }, [isPortalRoute, mode, setMode]);
  useEffect(() => {
    void recordUiEvent({ page: routeSegment || "home", action: "view" });
  }, [routeSegment]);
  const searchParams = useSearchParams();
  const navigationKey = `${pathname}?${searchParams.toString()}`;
  const [deploymentHealthHandle] = useState(createDeploymentHealthHandle);
  const isFullBleed = FULL_BLEED_SEGMENTS.has(routeSegment);
  // A manual toggle holds only for the route it was made on; full-bleed routes default to collapsed.
  const [sidebarOverride, setSidebarOverride] = useState<{ segment: string; collapsed: boolean } | null>(null);
  const sidebarCollapsed = sidebarOverride?.segment === routeSegment ? sidebarOverride.collapsed : isFullBleed;
  const setSidebarOpen = (open: boolean) => setSidebarOverride({ segment: routeSegment, collapsed: !open });

  const isGateway = mode === "ai-gateway" || mode === "customer-portal";

  // Non-gateway (agent control plane) mode keeps the original full-width Navbar,
  // which carries the account menu; the redesigned sidebar + header shell is
  // scoped to the ai-gateway dashboard. Chat and the public model hub are
  // separate routes that likewise keep the old Navbar.
  if (!isGateway) {
    return (
      <div className="flex h-screen flex-col overflow-hidden bg-background">
        <Navbar
          accessToken={accessToken}
          isPublicPage={false}
          deploymentHealth={<DeploymentHealthIcon handle={deploymentHealthHandle} />}
          deploymentHealthLink={<DeploymentHealthLink handle={deploymentHealthHandle} />}
        />
        <DeploymentHealthBanner handle={deploymentHealthHandle} />
        <DeploymentHealthPanel handle={deploymentHealthHandle} />
        <UserBanner accessToken={accessToken} />
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
    <LiteAdminFrame>
      <SidebarProvider
        open={!sidebarCollapsed}
        onOpenChange={setSidebarOpen}
        style={sidebarWidthStyle}
        className={cn("h-screen overflow-hidden bg-background max-md:h-dvh", sidebarProviderClassName)}
      >
        <CloseMobileSidebarOnNavigation navigationKey={navigationKey} />
        <AppSidebar />
        <CommandPaletteProvider>
          <SidebarInset className="min-w-0 overflow-hidden">
            <DashboardHeader
              accessToken={accessToken}
              navigationTrigger={<MobileSidebarTrigger label={t("shell.dashboardShell.openNavigation")} />}
              deploymentHealth={<DeploymentHealthIcon handle={deploymentHealthHandle} />}
              deploymentHealthLink={<DeploymentHealthLink handle={deploymentHealthHandle} />}
            />
            <DeploymentHealthBanner handle={deploymentHealthHandle} />
            <DeploymentHealthPanel handle={deploymentHealthHandle} />
            <UserBanner accessToken={accessToken} />
            <div
              className={
                routeSegment === "aria"
                  ? "flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden"
                  : "min-h-0 min-w-0 flex-1 overflow-y-auto"
              }
            >
              <TelemetryEnvBanner />
              {children}
            </div>
          </SidebarInset>
        </CommandPaletteProvider>
      </SidebarProvider>
    </LiteAdminFrame>
  );
}

function LayoutContent({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const pathname = usePathname();
  const { accessToken, token, authLoading, passwordResetRequired } = useAuth();
  const session = decodeToken(token);
  const { isLoading: currentUserLoading, isError: currentUserFailed } = useCurrentUser(
    session?.key ?? null,
    session?.auth_header_name,
  );
  const isInvitationFlow = Boolean(searchParams.get("invitation_id"));
  const identityPending = currentUserLoading || currentUserFailed;

  // Legacy invitation links point at /ui/?invitation_id=; the onboarding form now lives at its own
  // /onboarding route. Redirect once ui-config has loaded so uiHref resolves the SERVER_ROOT_PATH base.
  useEffect(() => {
    if (!authLoading && isInvitationFlow) {
      router.replace(`${uiHref("onboarding")}?${searchParams.toString()}`);
    }
  }, [authLoading, isInvitationFlow, router, searchParams]);

  // A session flagged for a forced password reset can only reach the change-password
  // endpoint server-side; keep the UI on the matching page.
  useEffect(() => {
    if (!authLoading && passwordResetRequired && !pathname?.endsWith("/change-password")) {
      router.replace(uiHref("change-password"));
    }
  }, [authLoading, passwordResetRequired, pathname, router]);

  useEffect(() => {
    if (!currentUserFailed || isInvitationFlow) return;
    clearTokenCookies();
    redirectToLogin();
  }, [currentUserFailed, isInvitationFlow]);

  if (authLoading || identityPending || isInvitationFlow) {
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
