import React from "react";
import {
  Table,
  TableHead,
  TableHeaderCell,
  TableBody,
  TableRow,
  TableCell,
} from "@tremor/react";
import { ChevronDownIcon, ChevronRightIcon } from "@heroicons/react/outline";
import { Button, Badge, Icon } from "@tremor/react";
import { Tooltip } from "antd";
import { TrashIcon, PencilAltIcon } from "@heroicons/react/outline";
import { getProviderLogoAndName } from "../provider_info_helpers";

interface TeamGroup {
  team_id: string;
  team_name: string;
  models: any[];
}

interface NestedModelTableProps {
  modelData: any[];
  expandedTeams: Set<string>;
  setExpandedTeams: (expandedTeams: Set<string>) => void;
  userRole: string;
  userID: string;
  premiumUser: boolean;
  setSelectedModelId: (id: string) => void;
  setSelectedTeamId: (id: string) => void;
  getDisplayModelName: (model: any) => string;
  handleEditClick: (model: any) => void;
  handleRefreshClick: () => void;
  setEditModel: (edit: boolean) => void;
  expandedRows: Set<string>;
  setExpandedRows: (expandedRows: Set<string>) => void;
}

export const NestedModelTable: React.FC<NestedModelTableProps> = ({
  modelData,
  expandedTeams,
  setExpandedTeams,
  userRole,
  userID,
  premiumUser,
  setSelectedModelId,
  setSelectedTeamId,
  getDisplayModelName,
  handleEditClick,
  handleRefreshClick,
  setEditModel,
  expandedRows,
  setExpandedRows,
}) => {
  // Transform flat model data into nested team structure
  const transformToNestedData = (models: any[]) => {
    const teamGroups: { [teamId: string]: TeamGroup } = {};
    
    // Group models by their accessible team IDs
    models.forEach(model => {
      const accessibleTeamIds = model.accesss_via_team_ids || [];
      
      // If model has no team access, put it in a default group
      if (accessibleTeamIds.length === 0) {
        const defaultTeamId = "unassigned";
        if (!teamGroups[defaultTeamId]) {
          teamGroups[defaultTeamId] = {
            team_id: defaultTeamId,
            team_name: "Unassigned Models",
            models: []
          };
        }
        teamGroups[defaultTeamId].models.push(model);
      } else {
        // Add model to each team it's accessible from
        accessibleTeamIds.forEach((teamId: string) => {
          if (!teamGroups[teamId]) {
            teamGroups[teamId] = {
              team_id: teamId,
              team_name: teamId, // You might want to fetch actual team names
              models: []
            };
          }
          teamGroups[teamId].models.push(model);
        });
      }
    });

    return Object.values(teamGroups);
  };

  const teamGroups = transformToNestedData(modelData);

  const toggleTeamExpansion = (teamId: string) => {
    const newExpandedTeams = new Set(expandedTeams);
    if (newExpandedTeams.has(teamId)) {
      newExpandedTeams.delete(teamId);
    } else {
      newExpandedTeams.add(teamId);
    }
    setExpandedTeams(newExpandedTeams);
  };

  const canEditModel = (model: any) => {
    return userRole === "Admin" || model.model_info?.created_by === userID;
  };

  return (
    <div className="rounded-lg custom-border relative">
      <div className="overflow-x-auto">
        <Table className="[&_td]:py-2 [&_th]:py-2 w-full">
          <TableHead>
            <TableRow>
              <TableHeaderCell className="py-1 h-8">Team / Model ID</TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">Public Model Name</TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">Provider</TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">LiteLLM Model Name</TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">Created At</TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">Updated At</TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">Created By</TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">
                <Tooltip title="Cost per 1M tokens">
                  <span>Input Cost</span>
                </Tooltip>
              </TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">
                <Tooltip title="Cost per 1M tokens">
                  <span>Output Cost</span>
                </Tooltip>
              </TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">Team ID</TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">Model Access Group</TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">Credentials</TableHeaderCell>
              <TableHeaderCell className="py-1 h-8">Status</TableHeaderCell>
              <TableHeaderCell className="py-1 h-8 sticky right-0 bg-white">Actions</TableHeaderCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {teamGroups.map((teamGroup) => (
              <React.Fragment key={teamGroup.team_id}>
                {/* Team Row */}
                <TableRow 
                  className="bg-gray-50 hover:bg-gray-100 cursor-pointer border-b-2 border-gray-200"
                  onClick={() => toggleTeamExpansion(teamGroup.team_id)}
                >
                  <TableCell className="py-3">
                    <div className="flex items-center gap-2">
                      {expandedTeams.has(teamGroup.team_id) ? (
                        <ChevronDownIcon className="h-4 w-4 text-gray-600" />
                      ) : (
                        <ChevronRightIcon className="h-4 w-4 text-gray-600" />
                      )}
                      <div className="font-semibold text-gray-800">
                        {teamGroup.team_name}
                      </div>
                      <Badge size="xs" color="blue" className="ml-2">
                        {teamGroup.models.length} models
                      </Badge>
                    </div>
                  </TableCell>
                  <TableCell colSpan={13} className="py-3">
                    <Button
                      size="xs"
                      variant="light"
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedTeamId(teamGroup.team_id);
                      }}
                    >
                      View Team Details
                    </Button>
                  </TableCell>
                </TableRow>

                {/* Model Rows (shown when team is expanded) */}
                {expandedTeams.has(teamGroup.team_id) && teamGroup.models.map((model) => (
                  <TableRow key={model.model_info.id} className="bg-white hover:bg-gray-50">
                    {/* Model ID */}
                    <TableCell className="py-2 pl-8">
                      <Tooltip title={model.model_info.id}>
                        <div 
                          className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left w-full truncate whitespace-nowrap cursor-pointer max-w-[15ch]"
                          onClick={() => setSelectedModelId(model.model_info.id)}
                        >
                          {model.model_info.id}
                        </div>
                      </Tooltip>
                    </TableCell>
                    
                    {/* Public Model Name */}
                    <TableCell className="py-2">
                      <Tooltip title={getDisplayModelName(model) || "-"}>
                        <div className="text-xs truncate whitespace-nowrap">
                          {getDisplayModelName(model) || "-"}
                        </div>
                      </Tooltip>
                    </TableCell>
                    
                    {/* Provider */}
                    <TableCell className="py-2">
                      <div className="flex items-center space-x-2">
                        {model.provider && (
                          <img
                            src={getProviderLogoAndName(model.provider).logo}
                            alt={`${model.provider} logo`}
                            className="w-4 h-4"
                            onError={(e) => {
                              const target = e.target as HTMLImageElement;
                              const parent = target.parentElement;
                              if (parent) {
                                const fallbackDiv = document.createElement('div');
                                fallbackDiv.className = 'w-4 h-4 rounded-full bg-gray-200 flex items-center justify-center text-xs';
                                fallbackDiv.textContent = model.provider?.charAt(0) || '-';
                                parent.replaceChild(fallbackDiv, target);
                              }
                            }}
                          />
                        )}
                        <p className="text-xs">{model.provider || "-"}</p>
                      </div>
                    </TableCell>
                    
                    {/* LiteLLM Model Name */}
                    <TableCell className="py-2">
                      <Tooltip title={model.litellm_model_name}>
                        <div className="text-xs truncate whitespace-nowrap">
                          {model.litellm_model_name || "-"}
                        </div>
                      </Tooltip>
                    </TableCell>
                    
                    {/* Created At */}
                    <TableCell className="py-2">
                      <span className="text-xs">
                        {model.model_info.created_at ? new Date(model.model_info.created_at).toLocaleDateString() : "-"}
                      </span>
                    </TableCell>
                    
                    {/* Updated At */}
                    <TableCell className="py-2">
                      <span className="text-xs">
                        {model.model_info.updated_at ? new Date(model.model_info.updated_at).toLocaleDateString() : "-"}
                      </span>
                    </TableCell>
                    
                    {/* Created By */}
                    <TableCell className="py-2">
                      <span className="text-xs">
                        {model.model_info.created_by || "-"}
                      </span>
                    </TableCell>
                    
                    {/* Input Cost */}
                    <TableCell className="py-2">
                      <pre className="text-xs">
                        {model.input_cost || "-"}
                      </pre>
                    </TableCell>
                    
                    {/* Output Cost */}
                    <TableCell className="py-2">
                      <pre className="text-xs">
                        {model.output_cost || "-"}
                      </pre>
                    </TableCell>
                    
                    {/* Team ID */}
                    <TableCell className="py-2">
                      {model.model_info.team_id ? (
                        <div className="overflow-hidden">
                          <Tooltip title={model.model_info.team_id}>
                            <Button
                              size="xs"
                              variant="light"
                              className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
                              onClick={() => setSelectedTeamId(model.model_info.team_id)}
                            >
                              {model.model_info.team_id.slice(0, 7)}...
                            </Button>
                          </Tooltip>
                        </div>
                      ) : (
                        "-"
                      )}
                    </TableCell>
                    
                    {/* Model Access Group */}
                    <TableCell className="py-2">
                      {(() => {
                        const accessGroups = model.model_info.access_groups;
                        
                        if (!accessGroups || accessGroups.length === 0) {
                          return "-";
                        }
                        
                        const modelId = model.model_info.id;
                        const isExpanded = expandedRows.has(modelId);
                        const shouldShowExpandButton = accessGroups.length > 1;
                        
                        const toggleExpanded = () => {
                          const newExpanded = new Set(expandedRows);
                          if (isExpanded) {
                            newExpanded.delete(modelId);
                          } else {
                            newExpanded.add(modelId);
                          }
                          setExpandedRows(newExpanded);
                        };
                        
                        return (
                          <div className="flex items-center gap-1 overflow-hidden">
                            <Badge
                              size="xs"
                              color="blue"
                              className="text-xs px-1.5 py-0.5 h-5 leading-tight flex-shrink-0"
                            >
                              {accessGroups[0]}
                            </Badge>
                            
                            {(isExpanded || (!shouldShowExpandButton && accessGroups.length === 2)) && 
                              accessGroups.slice(1).map((group: string, index: number) => (
                                <Badge
                                  key={index + 1}
                                  size="xs"
                                  color="blue"
                                  className="text-xs px-1.5 py-0.5 h-5 leading-tight flex-shrink-0"
                                >
                                  {group}
                                </Badge>
                              ))
                            }
                            
                            {shouldShowExpandButton && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  toggleExpanded();
                                }}
                                className="text-xs text-blue-600 hover:text-blue-800 px-1 py-0.5 rounded hover:bg-blue-50 h-5 leading-tight flex-shrink-0 whitespace-nowrap"
                              >
                                {isExpanded ? '−' : `+${accessGroups.length - 1}`}
                              </button>
                            )}
                          </div>
                        );
                      })()}
                    </TableCell>
                    
                    {/* Credentials */}
                    <TableCell className="py-2">
                      {model.litellm_params && model.litellm_params.litellm_credential_name ? (
                        <div className="overflow-hidden">
                          <Tooltip title={model.litellm_params.litellm_credential_name}>
                            {model.litellm_params.litellm_credential_name.slice(0, 7)}...
                          </Tooltip>
                        </div>
                      ) : (
                        <span className="text-gray-400">-</span>
                      )}
                    </TableCell>
                    
                    {/* Status */}
                    <TableCell className="py-2">
                      <div className={`
                        inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium
                        ${model.model_info.db_model 
                          ? 'bg-blue-50 text-blue-600' 
                          : 'bg-gray-100 text-gray-600'}
                      `}>
                        {model.model_info.db_model ? "DB Model" : "Config Model"}
                      </div>
                    </TableCell>
                    
                    {/* Actions */}
                    <TableCell className="py-2 sticky right-0 bg-white">
                      <div className="flex items-center justify-end gap-2 pr-4">
                        <Icon
                          icon={PencilAltIcon}
                          size="sm"
                          onClick={() => {
                            if (canEditModel(model)) {
                              setSelectedModelId(model.model_info.id);
                              setEditModel(true);
                            }
                          }}
                          className={!canEditModel(model) ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
                        />
                        <Icon
                          icon={TrashIcon}
                          size="sm"
                          onClick={() => {
                            if (canEditModel(model)) {
                              setSelectedModelId(model.model_info.id);
                              setEditModel(false);
                            }
                          }}
                          className={!canEditModel(model) ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
                        />
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </React.Fragment>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}; 