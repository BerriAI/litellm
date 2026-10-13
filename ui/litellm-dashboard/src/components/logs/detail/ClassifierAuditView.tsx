import CopyButton from "@/components/shared/CopyButton";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ReactNode } from "react";
import { JsonViewer } from "./JsonViewer";

interface ClassifierAuditViewProps {
  request: Record<string, unknown>;
  response: unknown;
}

export function ClassifierAuditView({ request, response }: ClassifierAuditViewProps) {
  return (
    <div className="mb-6 space-y-4">
      <AuditField title="Classifier input" value={request.classifier_input}>
        Provider request payload. A cached call or disabled message logging may have no capture.
      </AuditField>
      <AuditField title="Originating request, credentials masked" value={request.originating_request_masked}>
        Comparison only. This source request was not appended to the classifier input.
      </AuditField>
      <AuditField title="Classifier response" value={response}>
        The returned verdict and any explanation supplied by the classifier. Later routing rules may change the tier.
      </AuditField>
    </div>
  );
}

function AuditField({ title, value, children }: { title: string; value: unknown; children: ReactNode }) {
  const serialized = JSON.stringify(value);
  const truncated = serialized?.includes("litellm_truncated") ?? false;

  return (
    <Card size="sm" role="region" aria-label={title}>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {value != null && <CopyButton value={JSON.stringify(value, null, 2)} label={`Copy ${title}`} />}
      </CardHeader>
      <CardContent>
        <p className="mb-3 text-sm text-muted-foreground">{children}</p>
        {truncated && (
          <p role="status" className="mb-3 text-sm text-warning">
            This stored copy is truncated. The complete payload is unavailable from the configured log storage.
          </p>
        )}
        {value == null ? (
          <p className="text-sm text-muted-foreground">Not captured or message logging disabled</p>
        ) : (
          <JsonViewer data={value} mode="formatted" />
        )}
      </CardContent>
    </Card>
  );
}
