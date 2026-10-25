import { useLanguage } from "@/contexts/LanguageContext";
import CopyButton from "@/components/shared/CopyButton";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ReactNode } from "react";
import { JsonViewer } from "./JsonViewer";

interface ClassifierAuditViewProps {
  request: Record<string, unknown>;
  response: unknown;
}

export function ClassifierAuditView({ request, response }: ClassifierAuditViewProps) {
  const { t } = useLanguage();
  return (
    <div className="mb-6 space-y-4">
      <AuditField title={t("Classifier input")} value={request.classifier_input}>
        {t("Provider request payload. A cached call or disabled message logging may have no capture.")}
      </AuditField>
      <AuditField title={t("Originating request, credentials masked")} value={request.originating_request_masked}>
        {t("Comparison only. This source request was not appended to the classifier input.")}
      </AuditField>
      <AuditField title={t("Classifier response")} value={response}>
        {t(
          "The returned verdict and any explanation supplied by the classifier. Later routing rules may change the tier.",
        )}
      </AuditField>
    </div>
  );
}

function AuditField({ title, value, children }: { title: string; value: unknown; children: ReactNode }) {
  const { t } = useLanguage();
  const serialized = JSON.stringify(value);
  const truncated = serialized?.includes("litellm_truncated") ?? false;

  return (
    <Card size="sm" role="region" aria-label={title}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {value != null && (
          <CopyButton value={JSON.stringify(value, null, 2)} label={t("Copy {value1}", { value1: String(title) })} />
        )}
      </CardHeader>
      <CardContent>
        <p className="mb-3 text-sm text-muted-foreground">{children}</p>
        {truncated && (
          <p role="status" className="mb-3 text-sm text-warning">
            {t("This stored copy is truncated. The complete payload is unavailable from the configured log storage.")}
          </p>
        )}
        {value == null ? (
          <p className="text-sm text-muted-foreground">{t("Not captured or message logging disabled")}</p>
        ) : (
          <JsonViewer data={value} mode="formatted" />
        )}
      </CardContent>
    </Card>
  );
}
