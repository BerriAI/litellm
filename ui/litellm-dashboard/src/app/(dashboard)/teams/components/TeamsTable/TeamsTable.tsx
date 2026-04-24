import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { Pencil, Trash2 } from "lucide-react";
import React from "react";
import {
  type KeyResponse,
  Team,
} from "@/components/key_team_helpers/key_list";
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
      <TableHeader>
        <TableRow>
          <TableHead>Team Name</TableHead>
          <TableHead>Team ID</TableHead>
          <TableHead>Created</TableHead>
          <TableHead>Spend (USD)</TableHead>
          <TableHead>Budget (USD)</TableHead>
          <TableHead>Models</TableHead>
          <TableHead>Organization</TableHead>
          <TableHead>Your Role</TableHead>
          <TableHead>Info</TableHead>
        </TableRow>
      </TableHeader>

      <TableBody>
        {teams && teams.length > 0
          ? teams
              .filter((team) => {
                if (!currentOrg) return true;
                return team.organization_id === currentOrg.organization_id;
              })
              .sort(
                (a, b) =>
                  new Date(b.created_at).getTime() -
                  new Date(a.created_at).getTime(),
              )
              .map((team: Team) => (
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
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <button
                              type="button"
                              data-testid="team-id-cell"
                              onClick={() => setSelectedTeamId(team.team_id)}
                              className="font-mono text-blue-500 bg-blue-50 dark:bg-blue-950/30 hover:bg-blue-100 dark:hover:bg-blue-950/60 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px] rounded"
                            >
                              {team.team_id.slice(0, 7)}...
                            </button>
                          </TooltipTrigger>
                          <TooltipContent>{team.team_id}</TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </div>
                  </TableCell>
                  <TableCell
                    style={{
                      maxWidth: "4px",
                      whiteSpace: "pre-wrap",
                      overflow: "hidden",
                    }}
                  >
                    {team.created_at
                      ? new Date(team.created_at).toLocaleDateString()
                      : "N/A"}
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
                    {team["max_budget"] !== null &&
                    team["max_budget"] !== undefined
                      ? team["max_budget"]
                      : "No limit"}
                  </TableCell>
                  <ModelsCell team={team} />
                  <TableCell>{team.organization_id}</TableCell>
                  <YourRoleCell team={team} userId={userId} />
                  <TableCell>
                    <div className="text-sm">
                      {perTeamInfo &&
                        team.team_id &&
                        perTeamInfo[team.team_id] &&
                        perTeamInfo[team.team_id].keys &&
                        perTeamInfo[team.team_id].keys.length}{" "}
                      Keys
                    </div>
                    <div className="text-sm">
                      {perTeamInfo &&
                        team.team_id &&
                        perTeamInfo[team.team_id] &&
                        perTeamInfo[team.team_id].team_info &&
                        perTeamInfo[team.team_id].team_info
                          .members_with_roles &&
                        perTeamInfo[team.team_id].team_info.members_with_roles
                          .length}{" "}
                      Members
                    </div>
                  </TableCell>
                  <TableCell>
                    {userRole == "Admin" ? (
                      <div className="flex gap-1">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6"
                          onClick={() => {
                            setSelectedTeamId(team.team_id);
                            setEditTeam(true);
                          }}
                          aria-label="Edit team"
                        >
                          <Pencil className="h-3.5 w-3.5" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-6 w-6 text-destructive hover:text-destructive hover:bg-destructive/10"
                          onClick={() => onDeleteTeam(team.team_id)}
                          aria-label="Delete team"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </div>
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
