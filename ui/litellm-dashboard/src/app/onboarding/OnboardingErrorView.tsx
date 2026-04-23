import React from "react";
import { AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

export function OnboardingErrorView() {
  return (
    <div className="mx-auto w-full max-w-md mt-10">
      <div className="flex gap-2 items-start p-4 rounded-md border border-destructive/30 bg-destructive/10 text-destructive">
        <AlertCircle className="h-5 w-5 mt-0.5 shrink-0" />
        <div>
          <div className="font-semibold">Failed to load invitation</div>
          <div className="text-sm">
            The invitation link may be invalid or expired.
          </div>
        </div>
      </div>
      <div className="mt-4">
        <Button asChild variant="outline">
          <a href="/ui/login">Back to Login</a>
        </Button>
      </div>
    </div>
  );
}
