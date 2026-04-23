import React from "react";
import { Loader2 } from "lucide-react";

export function OnboardingLoadingView() {
  return (
    <div className="mx-auto w-full max-w-md mt-10 flex justify-center">
      <Loader2 className="h-8 w-8 animate-spin text-primary" />
    </div>
  );
}
