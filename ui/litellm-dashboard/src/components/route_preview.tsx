import React from "react";
import { ArrowRight, Info } from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

import { getProxyBaseUrl } from "./networking";

interface RoutePreviewProps {
  pathValue: string;
  targetValue: string;
  includeSubpath: boolean;
}

const Endpoint = ({ label, children }: { label: string; children: React.ReactNode }) => (
  <div className="min-w-0 flex-1 rounded-lg border bg-muted/40 p-3">
    <div className="mb-2 text-sm text-muted-foreground">{label}</div>
    <code className="block overflow-x-auto font-mono text-sm text-foreground">{children}</code>
  </div>
);

const RoutePreview: React.FC<RoutePreviewProps> = ({ pathValue, targetValue, includeSubpath }) => {
  const proxyBaseUrl = getProxyBaseUrl();

  if (!pathValue || !targetValue) {
    return null;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-lg">Route Preview</CardTitle>
        <CardDescription>How your requests will be routed</CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div>
          <h4 className="mb-3 text-base font-semibold">Basic routing:</h4>
          <div className="flex flex-col items-stretch gap-4 sm:flex-row sm:items-center">
            <Endpoint label="Your endpoint">{`${proxyBaseUrl}${pathValue}`}</Endpoint>
            <ArrowRight className="size-5 shrink-0 self-center text-muted-foreground max-sm:rotate-90" />
            <Endpoint label="Forwards to">{targetValue}</Endpoint>
          </div>
        </div>

        {includeSubpath ? (
          <div>
            <h4 className="mb-3 text-base font-semibold">With subpaths:</h4>
            <div className="flex flex-col items-stretch gap-4 sm:flex-row sm:items-center">
              <Endpoint label="Your endpoint + subpath">
                {`${proxyBaseUrl}${pathValue}`}
                <span className="text-primary">/v1/text-to-image/base/model</span>
              </Endpoint>
              <ArrowRight className="size-5 shrink-0 self-center text-muted-foreground max-sm:rotate-90" />
              <Endpoint label="Forwards to">
                {targetValue}
                <span className="text-primary">/v1/text-to-image/base/model</span>
              </Endpoint>
            </div>
            <p className="mt-3 text-sm text-muted-foreground">
              Any path after {pathValue} will be appended to the target URL
            </p>
          </div>
        ) : (
          <div className="flex items-start gap-2 rounded-md border border-primary/20 bg-primary/5 p-3 text-sm">
            <Info className="mt-0.5 size-4 shrink-0 text-primary" />
            <p>
              <span className="font-medium">Not seeing the routing you wanted?</span> Try enabling - Include Subpaths -
              above - this allows subroutes like{" "}
              <code className="rounded-sm bg-primary/10 px-1 py-0.5 font-mono text-xs">/api/v1/models</code> to be
              forwarded automatically.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default RoutePreview;
