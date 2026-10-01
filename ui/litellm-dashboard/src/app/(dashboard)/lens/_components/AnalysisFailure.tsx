"use client";

import { AlertTriangle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { failureParts } from "./lensData";

export function AnalysisFailure({ error }: { error: string }) {
  const { reason, fix } = failureParts(error);
  return (
    <Alert variant="destructive" className="border-destructive/30">
      <AlertTriangle aria-hidden />
      <AlertTitle>Analysis failed</AlertTitle>
      <AlertDescription className="space-y-1">
        <p>{reason}</p>
        {fix && (
          <p className="text-foreground">
            <span className="font-medium">Suggested fix: </span>
            {fix}
          </p>
        )}
      </AlertDescription>
    </Alert>
  );
}
