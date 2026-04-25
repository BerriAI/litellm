import { Badge, Icon, TableCell, Text } from "@tremor/react";
import { ChevronDownIcon, ChevronRightIcon } from "@heroicons/react/outline";
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

  const isAllModels = !team.models || team.models.length === 0 || team.models.includes("all-proxy-models");

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
        <Badge key={index} size={"xs"} color="red">
          <Text>All Proxy Models</Text>
        </Badge>
      );
    }
    const displayName = getModelDisplayName(entry.name);
    const truncated = displayName.length > 30 ? `${displayName.slice(0, 30)}...` : displayName;
    return (
      <Badge
        key={index}
        size={"xs"}
        color={entry.source === "access_group" ? "green" : "blue"}
        title={entry.source === "access_group" ? "From access group" : "Direct assignment"}
      >
        <Text>{truncated}</Text>
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
          <Badge size={"xs"} className="mb-1" color="red">
            <Text>All Proxy Models</Text>
          </Badge>
        ) : (
          <div className="flex flex-col">
            <div className="flex items-start">
              {modelEntries.length > 3 && (
                <div>
                  <Icon
                    icon={expandedAccordion ? ChevronDownIcon : ChevronRightIcon}
                    className="cursor-pointer"
                    size="xs"
                    onClick={() => {
                      setExpandedAccordion((prev) => !prev);
                    }}
                  />
                </div>
              )}
              <div className="flex flex-wrap gap-1">
                {modelEntries.slice(0, 3).map((entry, index) => renderBadge(entry, index))}
                {modelEntries.length > 3 && !expandedAccordion && (
                  <Badge size={"xs"} color="gray" className="cursor-pointer">
                    <Text>
                      +{modelEntries.length - 3} {modelEntries.length - 3 === 1 ? "more model" : "more models"}
                    </Text>
                  </Badge>
                )}
                {expandedAccordion && (
                  <div className="flex flex-wrap gap-1">
                    {modelEntries.slice(3).map((entry, index) => renderBadge(entry, index + 3))}
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
