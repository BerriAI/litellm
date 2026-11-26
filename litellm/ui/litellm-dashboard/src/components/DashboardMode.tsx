"use client";

import { useTranslation } from "react-i18next";
import { useEffect, useState, type ReactNode } from "react";
import { discoverControlPlane } from "@/lib/http/controlPlane";
import { ControlPlaneProvider } from "@/contexts/ControlPlaneContext";
import { setServerRootPath } from "@/lib/serverRootPath";
import { AuthProvider } from "@/contexts/AuthContext";

export default function DashboardMode({ children }: { children: ReactNode }) {
  const { t } = useTranslation("translation", { keyPrefix: "controlPlane" });
  const [mode, setMode] = useState<Awaited<ReturnType<typeof discoverControlPlane>> | Error | undefined>();

  useEffect(() => {
    let active = true;
    discoverControlPlane()
      .catch(() => new Error("Could not determine dashboard mode. Reload to retry."))
      .then((result) => {
        if (!active) return;
        if (result && !(result instanceof Error)) setServerRootPath(result.rootPath);
        setMode(result);
      });
    return () => {
      active = false;
    };
  }, []);

  if (mode === undefined)
    return (
      <main className="p-8" role="status">
        {t("loadingDashboard")}
      </main>
    );
  if (mode instanceof Error)
    return (
      <main className="p-8" role="alert">
        {mode.message}
      </main>
    );
  if (mode?.disabled) return <main className="p-8">{t("theAdminUiIsDisabled")}</main>;
  if (mode)
    return (
      <ControlPlaneProvider key={mode.prefix} prefix={mode.prefix} version={mode.version}>
        {children}
      </ControlPlaneProvider>
    );
  return <AuthProvider>{children}</AuthProvider>;
}
