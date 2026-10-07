"use client";

import { $api } from "@/lib/http/api";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import type { components } from "@/lib/http/schema";

interface KubernetesPodsCardProps {
  modelId: string;
}

interface KubernetesPodsCardContentProps {
  data: components["schemas"]["KubernetesPodsResponse"] | undefined;
  isError: boolean;
  isPending: boolean;
  apiErrorMessage: string;
}

function KubernetesPodsCardContent({ data, isError, isPending, apiErrorMessage }: KubernetesPodsCardContentProps) {
  const hasData = data !== undefined && !isError;
  const hasLookupError = hasData && data.error !== null;
  const showEmptyState = hasData && data.error === null && data.pod_ips.length === 0;
  const showPodIps = hasData && data.error === null && data.pod_ips.length > 0;
  const podCount = hasData && !hasLookupError ? data.pod_count : null;

  return (
    <>
      {podCount !== null && <p className="mt-4 text-sm">{podCount} ready pods</p>}
      {isPending && (
        <p className="mt-4 text-sm text-muted-foreground" role="status">
          Loading Kubernetes pods
        </p>
      )}
      {isError && (
        <p className="mt-4 text-sm text-destructive" role="alert">
          {apiErrorMessage}
        </p>
      )}
      {data && hasLookupError && (
        <p className="mt-4 text-sm text-destructive" role="alert">
          {data.error}
        </p>
      )}
      {data && showEmptyState && <p className="mt-4 text-sm text-muted-foreground">No ready pods</p>}
      {data && showPodIps && (
        <ul aria-label="Ready Kubernetes pod IPs" className="mt-3 space-y-1 font-mono text-sm">
          {data.pod_ips.map((ip) => (
            <li key={ip}>{ip}</li>
          ))}
        </ul>
      )}
    </>
  );
}

export default function KubernetesPodsCard({ modelId }: KubernetesPodsCardProps) {
  const { data, error, isError, isPending, refetch } = $api.useQuery(
    "get",
    "/model/{model_id}/kubernetes_pods",
    { params: { path: { model_id: modelId } } },
    { refetchInterval: 10_000 },
  );
  const apiErrorMessage =
    error instanceof Error ? error.message : JSON.stringify(error) ?? "Failed to load Kubernetes pods";

  return (
    <Card className="block p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 className="text-lg font-medium">Kubernetes pods</h3>
          {data && (
            <p className="mt-1 text-sm text-muted-foreground">
              {data.port === null ? data.service_host : `${data.service_host}:${data.port}`}
            </p>
          )}
        </div>
        <Button type="button" variant="outline" onClick={() => void refetch()}>
          Refresh
        </Button>
      </div>
      <KubernetesPodsCardContent
        data={data}
        isError={isError}
        isPending={isPending}
        apiErrorMessage={apiErrorMessage}
      />
    </Card>
  );
}
