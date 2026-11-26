"use client";

import { createContext, useCallback, useContext, useLayoutEffect, useMemo, useState, type ReactNode } from "react";
import {
  QueryClient,
  QueryClientProvider,
  useQueries,
  useQuery,
  useQueryClient,
  type UseQueryResult,
} from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { registerSameOriginProxyBase } from "@/components/networking";
import { registerControlPlaneProxy } from "@/lib/http/controlPlaneProxy";
import {
  controlPlaneApi,
  controlPlaneSessionKey,
  hasDataplane,
  hasScope,
  type ControlPlaneApi,
  type ControlPlanePrincipal,
} from "@/lib/http/controlPlane";
import { Button } from "@/components/ui/button";
import { ControlPlaneLogin } from "@/components/ControlPlaneLogin";
import { ControlPlaneAuthProvider } from "./AuthContext";
import { ErrorNotice } from "@/components/ControlPlaneShared";
import LoadingScreen from "@/components/common_components/LoadingScreen";

export type DataplanesQuery = UseQueryResult<Awaited<ReturnType<ControlPlaneApi["dataplanes"]>>>;

type Session = {
  version: string | null;
  api: ControlPlaneApi;
  principal: ControlPlanePrincipal;
  token: string;
  canWrite: boolean;
  logout: () => void;
  selected: string | null;
  options: { id: string; label: string }[];
  planes: DataplanesQuery;
  select: (id: string) => void;
};

const ControlPlaneContext = createContext<Session | null>(null);

export const useControlPlane = () => useContext(ControlPlaneContext);

const createClient = () => new QueryClient({ defaultOptions: { queries: { retry: false } } });

function RoutedDashboard({
  children,
  session,
  prefix,
  expire,
}: {
  children: ReactNode;
  session: Session;
  prefix: string;
  expire: () => void;
}) {
  const { canWrite, principal, selected, token } = session;
  const canRead = hasScope(principal, "proxy:read");
  const [ready, setReady] = useState(false);
  const [client] = useState(createClient);

  useLayoutEffect(() => {
    const restore = registerSameOriginProxyBase();
    const proxySession = { prefix, dataplane: selected, token, canRead, canWrite, expire };
    const unregister = registerControlPlaneProxy(proxySession);
    queueMicrotask(() => setReady(true));
    return () => {
      unregister();
      restore();
      void client.cancelQueries();
    };
  }, [canRead, canWrite, client, expire, prefix, selected, token]);

  if (!ready) return <LoadingScreen />;
  return (
    <ControlPlaneContext.Provider value={session}>
      <QueryClientProvider client={client}>
        <ControlPlaneAuthProvider
          accessToken={token}
          subject={principal.subject}
          email={principal.email ?? null}
          canWrite={canWrite}
        >
          {children}
        </ControlPlaneAuthProvider>
      </QueryClientProvider>
    </ControlPlaneContext.Provider>
  );
}

function Workspace({
  api,
  children,
  expire,
  logout,
  prefix,
  principal,
  token,
  version,
}: {
  api: ControlPlaneApi;
  children: ReactNode;
  expire: () => void;
  logout: () => void;
  prefix: string;
  principal: ControlPlanePrincipal;
  token: string;
  version: string | null;
}) {
  const { t } = useTranslation("translation", { keyPrefix: "controlPlane" });
  const [selection, select] = useState<string | null>(null);
  const dataplanesQuery = {
    queryKey: ["control-plane", "dataplanes"],
    queryFn: api.dataplanes,
    enabled: hasScope(principal, "dataplanes:read"),
    refetchInterval: 15000,
  };
  const planes = useQuery(dataplanesQuery);
  const visible = (planes.data?.items ?? []).filter((plane) => hasDataplane(principal, plane.id));
  const selected = visible.find((plane) => plane.id === selection)?.id ?? visible[0]?.id ?? null;
  const health = useQueries({
    queries: visible.map((plane) => ({
      queryKey: ["control-plane", "health", plane.id],
      queryFn: () => api.health(plane.id),
      refetchInterval: 15000,
    })),
  });
  const options = visible.map((plane, index) => ({
    id: plane.id,
    label: `${plane.name} (${health[index].data?.status ?? (health[index].isError ? t("healthUnavailable") : t("checkingHealth"))})`,
  }));
  const session = {
    api,
    principal,
    token,
    selected,
    options,
    planes,
    select,
    logout,
    version,
    canWrite: hasScope(principal, "proxy:write"),
  };

  if (planes.isLoading) return <LoadingScreen />;
  return (
    <RoutedDashboard key={selected} session={session} prefix={prefix} expire={expire}>
      {children}
    </RoutedDashboard>
  );
}

function SessionContent({
  prefix,
  children,
  version,
}: {
  prefix: string;
  children: ReactNode;
  version: string | null;
}) {
  const { t } = useTranslation("translation", { keyPrefix: "controlPlane" });
  const storageKey = controlPlaneSessionKey(prefix);
  const client = useQueryClient();
  const [token, setToken] = useState(() => sessionStorage.getItem(storageKey));
  const [expired, setExpired] = useState(false);
  const clearSession = useCallback(
    (didExpire: boolean) => {
      sessionStorage.removeItem(storageKey);
      setToken(null);
      setExpired(didExpire);
      client.clear();
    },
    [client, storageKey],
  );
  const logout = useCallback(() => clearSession(false), [clearSession]);
  const expire = useCallback(() => clearSession(true), [clearSession]);
  const api = useMemo(() => controlPlaneApi(prefix, token ?? undefined, expire), [expire, prefix, token]);
  const principal = useQuery({
    queryKey: ["control-plane", "principal", token !== null],
    queryFn: api.me,
    enabled: Boolean(token),
  });

  function login(value: string) {
    sessionStorage.setItem(storageKey, value);
    setExpired(false);
    setToken(value);
  }

  if (!token) return <ControlPlaneLogin prefix={prefix} expired={expired} onLogin={login} />;
  if (principal.isPending) return <LoadingScreen />;
  if (!principal.data) {
    return (
      <main className="grid max-w-md gap-3 p-8">
        <ErrorNotice error={principal.error} />
        <Button onClick={() => void principal.refetch()}>{t("retry")}</Button>
        <Button variant="outline" onClick={logout}>
          {t("signOut")}
        </Button>
      </main>
    );
  }
  return (
    <Workspace
      api={api}
      prefix={prefix}
      principal={principal.data}
      token={token}
      expire={expire}
      logout={logout}
      version={version}
    >
      {children}
    </Workspace>
  );
}

export function ControlPlaneProvider({
  prefix,
  children,
  version,
}: {
  prefix: string;
  children: ReactNode;
  version: string | null;
}) {
  const [client] = useState(createClient);
  return (
    <QueryClientProvider client={client}>
      <SessionContent prefix={prefix} version={version}>
        {children}
      </SessionContent>
    </QueryClientProvider>
  );
}
