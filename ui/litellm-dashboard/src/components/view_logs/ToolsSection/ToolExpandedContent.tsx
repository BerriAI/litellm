import { useState } from "react";
import {
  ToggleGroup,
  ToggleGroupItem,
} from "@/components/ui/toggle-group";
import { ParsedTool } from "./types";
import { FormattedToolView } from "./FormattedToolView";
import { JsonToolView } from "./JsonToolView";

type ViewMode = "formatted" | "json";

interface ToolExpandedContentProps {
  tool: ParsedTool;
}

export function ToolExpandedContent({ tool }: ToolExpandedContentProps) {
  const [viewMode, setViewMode] = useState<ViewMode>("formatted");

  return (
    <div>
      <div className="flex justify-between mb-3">
        <span className="text-xs text-muted-foreground">Description</span>
        <ToggleGroup
          type="single"
          size="sm"
          value={viewMode}
          onValueChange={(v) => {
            if (!v) return;
            setViewMode(v as ViewMode);
          }}
        >
          <ToggleGroupItem value="formatted">Formatted</ToggleGroupItem>
          <ToggleGroupItem value="json">JSON</ToggleGroupItem>
        </ToggleGroup>
      </div>

      {viewMode === "formatted" ? (
        <FormattedToolView tool={tool} />
      ) : (
        <JsonToolView tool={tool} />
      )}
    </div>
  );
}
