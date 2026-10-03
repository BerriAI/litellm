"use client";

import { uiHref } from "@/utils/uiHref";

export function ReadinessBanner({ activityReady }: { activityReady: boolean }) {
  return (
    <div role="status" className="flex flex-wrap items-center gap-2 border-y py-3 text-sm text-muted-foreground">
      {activityReady
        ? "Connect a worker to run new investigations. Saved results are still available."
        : "Recorded activity is not ready. Saved results are still available."}
      {!activityReady && (
        <a className="font-medium underline" href={uiHref("lens/?tab=traces")}>
          Check traces
        </a>
      )}
    </div>
  );
}
