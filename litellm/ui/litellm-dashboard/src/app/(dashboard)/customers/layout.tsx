"use client";

import type { ReactNode } from "react";
import { Trans, useTranslation } from "react-i18next";
import { AlertTriangle } from "lucide-react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { $api } from "@/lib/http/api";

const enterpriseUrl = "https://www.litellm.ai/enterprise";

export default function CustomersLayout({ children }: { children: ReactNode }) {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const { data: limits } = $api.useQuery(
    "get",
    "/management/v1/customer_portal/limits",
    {},
    {
      enabled: Boolean(accessToken),
    },
  );

  return (
    <>
      {limits?.limits_apply && (
        <div className="space-y-3 px-6 pt-4">
          <p className="text-xs text-muted-foreground">
            <Trans
              i18nKey="customerPortal.limits.byline"
              components={{
                enterprise: (
                  <a
                    className="underline underline-offset-2"
                    href={enterpriseUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                  />
                ),
              }}
            />
          </p>
          {limits.any_limit_reached && (
            <Alert variant="destructive" className="border-destructive/40 bg-destructive/5">
              <AlertTriangle />
              <AlertTitle>{t("customerPortal.limits.reachedTitle")}</AlertTitle>
              <AlertDescription>
                <p>{t("customerPortal.limits.reachedDescription")}</p>
                <a href={enterpriseUrl} target="_blank" rel="noopener noreferrer">
                  {t("customerPortal.limits.upgrade")}
                </a>
              </AlertDescription>
            </Alert>
          )}
        </div>
      )}
      {children}
    </>
  );
}
