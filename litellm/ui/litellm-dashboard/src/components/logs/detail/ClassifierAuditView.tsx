import CopyButton from "@/components/shared/CopyButton";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { JsonViewer } from "./JsonViewer";

interface ClassifierAuditViewProps {
  request: Record<string, unknown>;
  response: unknown;
}

export function ClassifierAuditView({ request, response }: ClassifierAuditViewProps) {
  const { t } = useTranslation();
  return (
    <div className="mb-6 space-y-4">
      <AuditField title={t("logs.detail.classifierAudit.input.title")} value={request.classifier_input}>
        {t("logs.detail.classifierAudit.input.description")}
      </AuditField>
      <AuditField title={t("logs.detail.classifierAudit.originating.title")} value={request.originating_request_masked}>
        {t("logs.detail.classifierAudit.originating.description")}
      </AuditField>
      <AuditField title={t("logs.detail.classifierAudit.response.title")} value={response}>
        {t("logs.detail.classifierAudit.response.description")}
      </AuditField>
    </div>
  );
}

function AuditField({ title, value, children }: { title: string; value: unknown; children: ReactNode }) {
  const { t } = useTranslation();
  const serialized = JSON.stringify(value);
  const truncated = serialized?.includes("litellm_truncated") ?? false;

  return (
    <Card size="sm" role="region" aria-label={title}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {value != null && (
          <CopyButton value={JSON.stringify(value, null, 2)} label={t("logs.detail.classifierAudit.copy", { title })} />
        )}
      </CardHeader>
      <CardContent>
        <p className="mb-3 text-sm text-muted-foreground">{children}</p>
        {truncated && (
          <p role="status" className="mb-3 text-sm text-warning">
            {t("logs.detail.classifierAudit.truncated")}
          </p>
        )}
        {value == null ? (
          <p className="text-sm text-muted-foreground">{t("logs.detail.classifierAudit.notCaptured")}</p>
        ) : (
          <JsonViewer data={value} mode="formatted" />
        )}
      </CardContent>
    </Card>
  );
}
