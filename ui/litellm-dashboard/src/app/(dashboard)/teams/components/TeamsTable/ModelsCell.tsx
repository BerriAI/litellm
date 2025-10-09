import { Badge, Icon, TableCell, Text } from "@tremor/react";
import { ChevronDownIcon, ChevronRightIcon } from "@heroicons/react/outline";
import { getModelDisplayName } from "@/components/key_team_helpers/fetch_available_models_team_key";
import React, { useState } from "react";
import { Team } from "@/components/key_team_helpers/key_list";

interface ModelsCellProps {
  team: Team;
}

const ModelsCell = ({ team }: ModelsCellProps) => {
  const [expandedAccordions, setExpandedAccordions] = useState<Record<string, boolean>>({});

  return (
    <TableCell
      style={{
        maxWidth: "8-x",
        whiteSpace: "pre-wrap",
        overflow: "hidden",
      }}
      className={team.models.length > 3 ? "px-0" : ""}
    >
      <div className="flex flex-col">
        {Array.isArray(team.models) ? (
          <div className="flex flex-col">
            {team.models.length === 0 ? (
              <Badge size={"xs"} className="mb-1" color="red">
                <Text>All Proxy Models</Text>
              </Badge>
            ) : (
              <>
                <div className="flex items-start">
                  {team.models.length > 3 && (
                    <div>
                      <Icon
                        icon={expandedAccordions[team.team_id] ? ChevronDownIcon : ChevronRightIcon}
                        className="cursor-pointer"
                        size="xs"
                        onClick={() => {
                          setExpandedAccordions((prev) => ({
                            ...prev,
                            [team.team_id]: !prev[team.team_id],
                          }));
                        }}
                      />
                    </div>
                  )}
                  <div className="flex flex-wrap gap-1">
                    {team.models.slice(0, 3).map((model: string, index: number) =>
                      model === "all-proxy-models" ? (
                        <Badge key={index} size={"xs"} color="red">
                          <Text>All Proxy Models</Text>
                        </Badge>
                      ) : (
                        <Badge key={index} size={"xs"} color="blue">
                          <Text>
                            {model.length > 30
                              ? `${getModelDisplayName(model).slice(0, 30)}...`
                              : getModelDisplayName(model)}
                          </Text>
                        </Badge>
                      ),
                    )}
                    {team.models.length > 3 && !expandedAccordions[team.team_id] && (
                      <Badge size={"xs"} color="gray" className="cursor-pointer">
                        <Text>
                          +{team.models.length - 3} {team.models.length - 3 === 1 ? "more model" : "more models"}
                        </Text>
                      </Badge>
                    )}
                    {expandedAccordions[team.team_id] && (
                      <div className="flex flex-wrap gap-1">
                        {team.models.slice(3).map((model: string, index: number) =>
                          model === "all-proxy-models" ? (
                            <Badge key={index + 3} size={"xs"} color="red">
                              <Text>All Proxy Models</Text>
                            </Badge>
                          ) : (
                            <Badge key={index + 3} size={"xs"} color="blue">
                              <Text>
                                {model.length > 30
                                  ? `${getModelDisplayName(model).slice(0, 30)}...`
                                  : getModelDisplayName(model)}
                              </Text>
                            </Badge>
                          ),
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        ) : null}
      </div>
    </TableCell>
  );
};

export default ModelsCell;
