"use client";

import type { Lens } from "../model/types";

export function WatchAllBanner({
  lenses,
  busy,
  onWatchAll,
  skipped = [],
}: {
  lenses: readonly Lens[];
  busy: boolean;
  onWatchAll: () => void;
  skipped?: readonly { id: string; name: string; reason: string }[];
}) {
  const paused = lenses.filter((lens) => !lens.settings.enabled).length;
  if (paused === 0) return null;
  if (skipped.length >= paused)
    return (
      <span
        role="status"
        className="text-xs text-muted-foreground"
        title={skipped.map((s) => `${s.name}: ${s.reason}`).join("\n")}
      >
        {paused} paused · {skipped.length === 1 ? `${skipped[0].name} needs a fix` : `${skipped.length} need a fix`}
      </span>
    );
  return (
    <span role="status" className="text-xs text-muted-foreground">
      {paused} paused ·{" "}
      <button
        type="button"
        disabled={busy}
        onClick={onWatchAll}
        className="font-medium text-foreground underline-offset-2 hover:underline disabled:opacity-50"
      >
        Turn all on
      </button>
    </span>
  );
}
