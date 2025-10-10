import {
  Button,
  Icon,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Text,
} from "@tremor/react";
import { Tooltip } from "antd";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { PencilAltIcon, TrashIcon } from "@heroicons/react/outline";
import React from "react";
import { type KeyResponse, Team } from "@/components/key_team_helpers/key_list";
import { Member, Organization } from "@/components/networking";
import ModelsCell from "@/app/(dashboard)/teams/components/TeamsTable/ModelsCell";
import YourRoleCell from "@/app/(dashboard)/teams/components/TeamsTable/YourRoleCell/YourRoleCell";

type TeamsTableProps = {
  teams: Team[] | null;
  currentOrg: Organization | null;
  perTeamInfo: Record<string, PerTeamInfo>;
  userRole: string | null;
  userId: string | null;
  setSelectedTeamId: (teamId: string) => void;
  setEditTeam: (editTeam: boolean) => void;
  onDeleteTeam: (teamId: string) => void;
};

interface TeamInfo {
  members_with_roles: Member[];
}

interface PerTeamInfo {
  keys: KeyResponse[];
  team_info: TeamInfo;
}

const TeamsTable = ({
  teams,
  currentOrg,
  setSelectedTeamId,
  perTeamInfo,
  userRole,
  userId,
  setEditTeam,
  onDeleteTeam,
}: TeamsTableProps) => {
  return (
    <Table>
      <TableHead>
        <TableRow>
          <TableHeaderCell>Team Name</TableHeaderCell>
          <TableHeaderCell>Team ID</TableHeaderCell>
          <TableHeaderCell>Created</TableHeaderCell>
          <TableHeaderCell>Spend (USD)</TableHeaderCell>
          <TableHeaderCell>Budget (USD)</TableHeaderCell>
          <TableHeaderCell>Models</TableHeaderCell>
          <TableHeaderCell>Organization</TableHeaderCell>
          <TableHeaderCell>Your Role</TableHeaderCell>
          <TableHeaderCell>Info</TableHeaderCell>
        </TableRow>
      </TableHead>

      <TableBody>
        {teams && teams.length > 0
          ? teams
              .filter((team) => {
                if (!currentOrg) return true;
                return team.organization_id === currentOrg.organization_id;
              })
              .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
              .map((team: any) => (
                <TableRow key={team.team_id}>
                  <TableCell
                    style={{
                      maxWidth: "4px",
                      whiteSpace: "pre-wrap",
                      overflow: "hidden",
                    }}
                  >
                    {team["team_alias"]}
                  </TableCell>
                  <TableCell>
                    <div className="overflow-hidden">
                      <Tooltip title={team.team_id}>
                        <Button
                          size="xs"
                          variant="light"
                          className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
                          onClick={() => {
                            // Add click handler
                            setSelectedTeamId(team.team_id);
                          }}
                        >
                          {team.team_id.slice(0, 7)}...
                        </Button>
                      </Tooltip>
                    </div>
                  </TableCell>
                  <TableCell
                    style={{
                      maxWidth: "4px",
                      whiteSpace: "pre-wrap",
                      overflow: "hidden",
                    }}
                  >
                    {team.created_at ? new Date(team.created_at).toLocaleDateString() : "N/A"}
                  </TableCell>
                  <TableCell
                    style={{
                      maxWidth: "4px",
                      whiteSpace: "pre-wrap",
                      overflow: "hidden",
                    }}
                  >
                    {formatNumberWithCommas(team["spend"], 4)}
                  </TableCell>
                  <TableCell
                    style={{
                      maxWidth: "4px",
                      whiteSpace: "pre-wrap",
                      overflow: "hidden",
                    }}
                  >
                    {team["max_budget"] !== null && team["max_budget"] !== undefined ? team["max_budget"] : "No limit"}
                  </TableCell>
                  <ModelsCell team={team} />
                  <TableCell>{team.organization_id}</TableCell>
                  <YourRoleCell team={team} userId={userId} />
                  <TableCell>
                    <Text>
                      {perTeamInfo &&
                        team.team_id &&
                        perTeamInfo[team.team_id] &&
                        perTeamInfo[team.team_id].keys &&
                        perTeamInfo[team.team_id].keys.length}{" "}
                      Keys
                    </Text>
                    <Text>
                      {perTeamInfo &&
                        team.team_id &&
                        perTeamInfo[team.team_id] &&
                        perTeamInfo[team.team_id].team_info &&
                        perTeamInfo[team.team_id].team_info.members_with_roles &&
                        perTeamInfo[team.team_id].team_info.members_with_roles.length}{" "}
                      Members
                    </Text>
                  </TableCell>
                  <TableCell>
                    {userRole == "Admin" ? (
                      <>
                        <Icon
                          icon={PencilAltIcon}
                          size="sm"
                          onClick={() => {
                            setSelectedTeamId(team.team_id);
                            setEditTeam(true);
                          }}
                        />
                        <Icon onClick={() => onDeleteTeam(team.team_id)} icon={TrashIcon} size="sm" />
                      </>
                    ) : null}
                  </TableCell>
                </TableRow>
              ))
          : null}
      </TableBody>
    </Table>
  );
};

export default TeamsTable;
