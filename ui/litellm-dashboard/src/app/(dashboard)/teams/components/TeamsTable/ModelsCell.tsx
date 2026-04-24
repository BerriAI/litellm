import { Badge } from "@/components/ui/badge";
import { TableCell } from "@/components/ui/table";
import { ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { getModelDisplayName } from "@/components/key_team_helpers/fetch_available_models_team_key";
import React, { useMemo, useState } from "react";
import { Team } from "@/components/key_team_helpers/key_list";

interface ModelsCellProps {
  team: Team;
}

interface ModelEntry {
  name: string;
  source: "direct" | "access_group";
}

const ModelsCell = ({ team }: ModelsCellProps) => {
  const [expandedAccordion, setExpandedAccordion] = useState<boolean>(false);

  const isAllModels =
    !team.models ||
    team.models.length === 0 ||
    team.models.includes("all-proxy-models");

  const modelEntries: ModelEntry[] = useMemo(() => {
    if (isAllModels) return [];
    const entries: ModelEntry[] = team.models.map((m) => ({
      name: m,
      source: "direct" as const,
    }));
    for (const m of team.access_group_models || []) {
      entries.push({ name: m, source: "access_group" });
    }
    return entries;
  }, [team.models, team.access_group_models, isAllModels]);

  const renderBadge = (entry: ModelEntry, index: number) => {
    if (entry.name === "all-proxy-models") {
      return (
        <Badge
          key={index}
          className="bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300 text-xs"
        >
          All Proxy Models
        </Badge>
      );
    }
    const displayName = getModelDisplayName(entry.name);
    const truncated =
      displayName.length > 30 ? `${displayName.slice(0, 30)}...` : displayName;
    return (
      <Badge
        key={index}
        className={cn(
          "text-xs",
          entry.source === "access_group"
            ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300"
            : "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
        )}
        title={
          entry.source === "access_group"
            ? "From access group"
            : "Direct assignment"
        }
      >
        {truncated}
      </Badge>
    );
  };

  return (
    <TableCell
      style={{
        maxWidth: "8-x",
        whiteSpace: "pre-wrap",
        overflow: "hidden",
      }}
      className={modelEntries.length > 3 ? "px-0" : ""}
    >
      <div className="flex flex-col">
        {modelEntries.length === 0 ? (
          <Badge className="bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300 text-xs mb-1">
            All Proxy Models
          </Badge>
        ) : (
          <div className="flex flex-col">
            <div className="flex items-start">
              {modelEntries.length > 3 && (
                <button
                  type="button"
                  onClick={() => setExpandedAccordion((prev) => !prev)}
                  className="cursor-pointer text-muted-foreground hover:text-foreground"
                  aria-label={expandedAccordion ? "Collapse" : "Expand"}
                >
                  {expandedAccordion ? (
                    <ChevronDown className="h-3 w-3" />
                  ) : (
                    <ChevronRight className="h-3 w-3" />
                  )}
                </button>
              )}
              <div className="flex flex-wrap gap-1">
                {modelEntries
                  .slice(0, 3)
                  .map((entry, index) => renderBadge(entry, index))}
                {modelEntries.length > 3 && !expandedAccordion && (
                  <Badge
                    variant="secondary"
                    className="text-xs cursor-pointer"
                  >
                    +{modelEntries.length - 3}{" "}
                    {modelEntries.length - 3 === 1
                      ? "more model"
                      : "more models"}
                  </Badge>
                )}
                {expandedAccordion && (
                  <div className="flex flex-wrap gap-1">
                    {modelEntries
                      .slice(3)
                      .map((entry, index) => renderBadge(entry, index + 3))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </TableCell>
  );
};

export default ModelsCell;
