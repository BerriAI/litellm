"use client";

import type { ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { createSerializer, useQueryStates } from "nuqs";
import { useTranslation } from "react-i18next";
import { ErrorState } from "@/components/shared/ErrorState";
import { fetchClient } from "@/lib/http/api";
import { extractProxyErrorMessage } from "@/lib/http/client";
import { uiHref } from "@/utils/uiHref";
import { isProxyAdminRole } from "@/utils/roles";
import useAuthorized from "../../hooks/useAuthorized";
import { customerAppParsers } from "../../customers/_components/CustomerHierarchyNavigation";
import {
  CustomerDetailHeader,
  CustomerDetailLayout,
  CustomerDetailNavigation,
} from "../../customers/_components/CustomerDetailLayout";
import { customerUsageDateParsers } from "../../customers/_components/customerUsageRange";
import type { App } from "./appForms";

const detailParsers = { ...customerAppParsers, ...customerUsageDateParsers };
const detailHref = createSerializer(detailParsers);
const pages = [
  ["usage", "customerPortal.hierarchy.usage"],
  ["budget-groups", "customerPortal.budgetGroups.title"],
  ["settings", "customerPortal.hierarchy.settings"],
] as const;

export default function AppDetailPage({
  active,
  children,
}: {
  active: (typeof pages)[number][0];
  children: (app: App) => ReactNode;
}) {
  const { t } = useTranslation();
  const [params] = useQueryStates(detailParsers);
  const { accessToken, userRole, isViewOnly, isLoading } = useAuthorized();
  const canEdit = isProxyAdminRole(userRole) && !isViewOnly;
  const denied = active !== "usage" && !canEdit;
  const app = useQuery({
    queryKey: ["app", params.app_id],
    queryFn: async () => {
      const response = await fetchClient.GET("/management/v1/apps/{app_id}", {
        params: { path: { app_id: params.app_id! } },
      });
      if (!response.data) throw new Error(t("apps.loadFailed"));
      return response.data;
    },
    enabled: Boolean(params.app_id && accessToken) && !denied,
  });
  const content = () => {
    if (isLoading) return <p role="status">{t("apps.loading")}</p>;
    if (denied) return <ErrorState message={t("customerPortal.hierarchy.adminOnly")} />;
    if (app.isError)
      return <ErrorState message={extractProxyErrorMessage(app.error)} onRetry={() => void app.refetch()} />;
    if (app.data) return children(app.data);
    return <p role="status">{t("apps.loading")}</p>;
  };
  return (
    <CustomerDetailLayout>
      <CustomerDetailHeader
        title={app.data?.alias ?? params.app_id ?? t("apps.selectApp")}
        breadcrumbLabel={t("ui.breadcrumb.label")}
        breadcrumbs={
          <a className="hover:text-foreground" href={uiHref("apps")}>
            {t("apps.list.back")}
          </a>
        }
      />
      {params.app_id && (
        <>
          <CustomerDetailNavigation
            label={t("apps.navigation")}
            items={pages
              .filter(([page]) => page === "usage" || canEdit)
              .map(([page, label]) => ({
                href: detailHref(uiHref(`apps/${page}`), params),
                label: t(label),
                active: active === page,
              }))}
          />
          {content()}
        </>
      )}
    </CustomerDetailLayout>
  );
}
