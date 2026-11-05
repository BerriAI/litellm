"use client";

import { useState } from "react";
import { Badge } from "@/components/ui/badge";

export default function UsageTags({
  tags,
  loading,
  failed,
}: {
  tags: readonly string[];
  loading: boolean;
  failed: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  if (loading) return <span className="text-xs text-muted-foreground">Loading tags…</span>;
  if (failed) return <span className="text-xs text-muted-foreground">Tags unavailable</span>;
  if (tags.length === 0) return <span className="text-xs text-muted-foreground">No recorded tags</span>;
  const visible = expanded ? tags : tags.slice(0, 5);
  return (
    <div className="flex max-w-72 flex-wrap items-center gap-1">
      {visible.map((tag) => (
        <Badge key={tag} variant="secondary" className="max-w-full whitespace-normal break-all font-normal">
          {tag}
        </Badge>
      ))}
      {tags.length > 5 && (
        <button
          type="button"
          className="text-xs text-primary hover:underline"
          aria-expanded={expanded}
          aria-label={expanded ? "Show fewer usage tags" : `Show ${tags.length - 5} more usage tags`}
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? "Show less" : `+${tags.length - 5} more`}
        </button>
      )}
    </div>
  );
}
