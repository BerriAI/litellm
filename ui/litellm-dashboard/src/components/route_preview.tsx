import React from "react";
import { Card } from "@/components/ui/card";
import { ChevronRight, Info } from "lucide-react";
import { getProxyBaseUrl } from "./networking";

interface RoutePreviewProps {
  pathValue: string;
  targetValue: string;
  includeSubpath: boolean;
}

const RoutePreview: React.FC<RoutePreviewProps> = ({
  pathValue,
  targetValue,
  includeSubpath,
}) => {
  const proxyBaseUrl = getProxyBaseUrl();

  const getLiteLLMProxyUrl = () =>
    pathValue ? `${proxyBaseUrl}${pathValue}` : "";

  if (!pathValue || !targetValue) {
    return null;
  }

  return (
    <Card className="p-5">
      <h5 className="text-lg font-semibold text-foreground mb-2">
        Route Preview
      </h5>
      <p className="text-muted-foreground mb-5 text-sm">
        How your requests will be routed
      </p>

      <div className="space-y-5">
        <div>
          <div className="text-base font-semibold text-foreground mb-3">
            Basic routing:
          </div>
          <div className="flex items-center gap-4">
            <div className="flex-1 bg-muted border border-border rounded-lg p-3">
              <div className="text-sm text-muted-foreground mb-2">
                Your endpoint
              </div>
              <code className="font-mono text-sm text-foreground">
                {getLiteLLMProxyUrl()}
              </code>
            </div>

            <div className="text-muted-foreground">
              <ChevronRight className="h-5 w-5" />
            </div>

            <div className="flex-1 bg-muted border border-border rounded-lg p-3">
              <div className="text-sm text-muted-foreground mb-2">
                Forwards to
              </div>
              <code className="font-mono text-sm text-foreground">
                {targetValue}
              </code>
            </div>
          </div>
        </div>

        {includeSubpath && (
          <div>
            <div className="text-base font-semibold text-foreground mb-3">
              With subpaths:
            </div>
            <div className="flex items-center gap-4">
              <div className="flex-1 bg-muted border border-border rounded-lg p-3">
                <div className="text-sm text-muted-foreground mb-2">
                  Your endpoint + subpath
                </div>
                <code className="font-mono text-sm text-foreground">
                  {pathValue && `${proxyBaseUrl}${pathValue}`}
                  <span className="text-primary">
                    /v1/text-to-image/base/model
                  </span>
                </code>
              </div>

              <div className="text-muted-foreground">
                <ChevronRight className="h-5 w-5" />
              </div>

              <div className="flex-1 bg-muted border border-border rounded-lg p-3">
                <div className="text-sm text-muted-foreground mb-2">
                  Forwards to
                </div>
                <code className="font-mono text-sm text-foreground">
                  {targetValue}
                  <span className="text-primary">
                    /v1/text-to-image/base/model
                  </span>
                </code>
              </div>
            </div>

            <div className="mt-3 text-sm text-muted-foreground">
              Any path after {pathValue} will be appended to the target URL
            </div>
          </div>
        )}

        {!includeSubpath && (
          <div className="mt-4 p-3 bg-primary/5 rounded-md border border-primary/20">
            <div className="flex items-start">
              <Info className="text-primary mt-0.5 mr-2 h-4 w-4 flex-shrink-0" />
              <div className="text-sm text-primary">
                <span className="font-medium">
                  Not seeing the routing you wanted?
                </span>{" "}
                Try enabling - Include Subpaths - above - this allows subroutes
                like{" "}
                <code className="bg-primary/10 px-1 py-0.5 rounded font-mono text-xs">
                  /api/v1/models
                </code>{" "}
                to be forwarded automatically.
              </div>
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default RoutePreview;
