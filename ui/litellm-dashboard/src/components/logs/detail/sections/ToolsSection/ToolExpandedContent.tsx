/**
 * Expanded content for a tool with view mode toggle
 */

import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ParsedTool } from "./types";
import { FormattedToolView } from "./FormattedToolView";
import { JsonToolView } from "./JsonToolView";

type ViewMode = "formatted" | "json";

interface ToolExpandedContentProps {
  tool: ParsedTool;
}

export function ToolExpandedContent({ tool }: ToolExpandedContentProps) {
  const { t } = useTranslation();
  const [viewMode, setViewMode] = useState<ViewMode>("formatted");

  return (
    <div>
      {/* View Mode Toggle - Top Right */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginBottom: 12,
        }}
      >
        <span className="text-xs text-muted-foreground">{t("logs.detail.tools.description")}</span>
        <Tabs value={viewMode} onValueChange={(value) => setViewMode(value as ViewMode)}>
          <TabsList>
            <TabsTrigger value="formatted">{t("logs.detail.tools.formatted")}</TabsTrigger>
            <TabsTrigger value="json">{t("logs.detail.tools.json")}</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {viewMode === "formatted" ? <FormattedToolView tool={tool} /> : <JsonToolView tool={tool} />}
    </div>
  );
}
