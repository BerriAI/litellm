import React, { useState, useEffect } from "react";
import { teamDeleteCall, Organization } from "@/components/networking";
import { fetchTeams } from "@/components/common_components/fetch_teams";
import { Form } from "antd";
import TeamInfoView from "@/components/team/team_info";
import TeamSSOSettings from "@/components/TeamSSOSettings";
import { isAdminRole } from "@/utils/roles";
import { Card, Button, Col, Text, Grid, TabPanel } from "@tremor/react";
import AvailableTeamsPanel from "@/components/team/available_teams";
import type { KeyResponse, Team } from "@/components/key_team_helpers/key_list";

import { Member, v2TeamListCall } from "@/components/networking";
import { updateExistingKeys } from "@/utils/dataUtils";
import TeamsHeaderTabs from "@/app/(dashboard)/teams/components/TeamsHeaderTabs";
import TeamsFilters from "@/app/(dashboard)/teams/components/TeamsFilters";
import useFetchTeams from "@/app/(dashboard)/teams/hooks/useFetchTeams";
import TeamsTable from "@/app/(dashboard)/teams/components/TeamsTable/TeamsTable";
import DeleteTeamModal from "@/app/(dashboard)/teams/components/modals/DeleteTeamModal";
import CreateTeamModal from "@/app/(dashboard)/teams/components/modals/CreateTeamModal";

interface TeamProps {
  teams: Team[] | null;
  accessToken: string | null;
  setTeams: React.Dispatch<React.SetStateAction<Team[] | null>>;
  userID: string | null;
  userRole: string | null;
  organizations: Organization[] | null;
  premiumUser?: boolean;
}

interface FilterState {
  team_id: string;
  team_alias: string;
  organization_id: string;
  sort_by: string;
  sort_order: "asc" | "desc";
}

interface TeamInfo {
  members_with_roles: Member[];
}

interface PerTeamInfo {
  keys: KeyResponse[];
  team_info: TeamInfo;
}

const TeamsView: React.FC<TeamProps> = ({
  teams,
  accessToken,
  setTeams,
  userID,
  userRole,
  organizations,
  premiumUser = false,
}) => {
  const [currentOrg, setCurrentOrg] = useState<Organization | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState<FilterState>({
    team_id: "",
    team_alias: "",
    organization_id: "",
    sort_by: "created_at",
    sort_order: "desc",
  });

  const [form] = Form.useForm();
  const [memberForm] = Form.useForm();

  const [selectedTeamId, setSelectedTeamId] = useState<string | null>(null);
  const [editTeam, setEditTeam] = useState<boolean>(false);

  const [isTeamModalVisible, setIsTeamModalVisible] = useState(false);
  const [isAddMemberModalVisible, setIsAddMemberModalVisible] = useState(false);
  const [isEditMemberModalVisible, setIsEditMemberModalVisible] = useState(false);
  const [userModels, setUserModels] = useState<string[]>([]);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [teamToDelete, setTeamToDelete] = useState<string | null>(null);
  const [perTeamInfo, setPerTeamInfo] = useState<Record<string, PerTeamInfo>>({});

  const [loggingSettings, setLoggingSettings] = useState<any[]>([]);
  const [modelAliases, setModelAliases] = useState<{ [key: string]: string }>({});
  const { lastRefreshed, onRefreshClick: handleRefreshClick } = useFetchTeams({ currentOrg, setTeams });

  useEffect(() => {
    const fetchTeamInfo = () => {
      if (!teams) return;

      const newPerTeamInfo = teams.reduce(
        (acc, team) => {
          acc[team.team_id] = {
            keys: team.keys || [],
            team_info: {
              members_with_roles: team.members_with_roles || [],
            },
          };
          return acc;
        },
        {} as Record<string, PerTeamInfo>,
      );

      setPerTeamInfo(newPerTeamInfo);
    };

    fetchTeamInfo();
  }, [teams]);

  const handleOk = () => {
    setIsTeamModalVisible(false);
    form.resetFields();
    setLoggingSettings([]);
    setModelAliases({});
  };

  const handleMemberOk = () => {
    setIsAddMemberModalVisible(false);
    setIsEditMemberModalVisible(false);
    memberForm.resetFields();
  };

  const handleCancel = () => {
    setIsTeamModalVisible(false);
    form.resetFields();
    setLoggingSettings([]);
    setModelAliases({});
  };

  const handleDelete = async (team_id: string) => {
    // Set the team to delete and open the confirmation modal
    setTeamToDelete(team_id);
    setIsDeleteModalOpen(true);
  };

  const confirmDelete = async () => {
    if (teamToDelete == null || teams == null || accessToken == null) {
      return;
    }

    try {
      await teamDeleteCall(accessToken, teamToDelete);
      // Successfully completed the deletion. Update the state to trigger a rerender.
      fetchTeams(accessToken, userID, userRole, currentOrg, setTeams);
    } catch (error) {
      console.error("Error deleting the team:", error);
      // Handle any error situations, such as displaying an error message to the user.
    }

    // Close the confirmation modal and reset the teamToDelete
    setIsDeleteModalOpen(false);
    setTeamToDelete(null);
  };

  const cancelDelete = () => {
    // Close the confirmation modal and reset the teamToDelete
    setIsDeleteModalOpen(false);
    setTeamToDelete(null);
  };

  const is_team_admin = (team: any) => {
    if (team == null || team.members_with_roles == null) {
      return false;
    }
    for (let i = 0; i < team.members_with_roles.length; i++) {
      let member = team.members_with_roles[i];
      if (member.user_id == userID && member.role == "admin") {
        return true;
      }
    }
    return false;
  };

  const handleFilterChange = (key: keyof FilterState, value: string) => {
    const newFilters = { ...filters, [key]: value };
    setFilters(newFilters);
    // Call teamListCall with the new filters
    if (accessToken) {
      v2TeamListCall(
        accessToken,
        newFilters.organization_id || null,
        null,
        newFilters.team_id || null,
        newFilters.team_alias || null,
      )
        .then((response) => {
          if (response && response.teams) {
            setTeams(response.teams);
          }
        })
        .catch((error) => {
          console.error("Error fetching teams:", error);
        });
    }
  };

  const handleSortChange = (sortBy: string, sortOrder: "asc" | "desc") => {
    const newFilters = {
      ...filters,
      sort_by: sortBy,
      sort_order: sortOrder,
    };
    setFilters(newFilters);
    // Call teamListCall with the new sort parameters
    if (accessToken) {
      v2TeamListCall(
        accessToken,
        filters.organization_id || null,
        null,
        filters.team_id || null,
        filters.team_alias || null,
      )
        .then((response) => {
          if (response && response.teams) {
            setTeams(response.teams);
          }
        })
        .catch((error) => {
          console.error("Error fetching teams:", error);
        });
    }
  };

  const handleFilterReset = () => {
    setFilters({
      team_id: "",
      team_alias: "",
      organization_id: "",
      sort_by: "created_at",
      sort_order: "desc",
    });
    // Reset teams list
    if (accessToken) {
      v2TeamListCall(accessToken, null, userID || null, null, null)
        .then((response) => {
          if (response && response.teams) {
            setTeams(response.teams);
          }
        })
        .catch((error) => {
          console.error("Error fetching teams:", error);
        });
    }
  };

  return (
    <div className="w-full mx-4 h-[75vh]">
      <Grid numItems={1} className="gap-2 p-8 w-full mt-2">
        <Col numColSpan={1} className="flex flex-col gap-2">
          {(userRole == "Admin" || userRole == "Org Admin") && (
            <Button className="w-fit" onClick={() => setIsTeamModalVisible(true)}>
              + Create New Team
            </Button>
          )}
          {selectedTeamId ? (
            <TeamInfoView
              teamId={selectedTeamId}
              onUpdate={(data) => {
                setTeams((teams) => {
                  if (teams == null) {
                    return teams;
                  }
                  const updated = teams.map((team) => {
                    if (data.team_id === team.team_id) {
                      return updateExistingKeys(team, data);
                    }
                    return team;
                  });
                  // Minimal fix: refresh the full team list after an update
                  if (accessToken) {
                    fetchTeams(accessToken, userID, userRole, currentOrg, setTeams);
                  }
                  return updated;
                });
              }}
              onClose={() => {
                setSelectedTeamId(null);
                setEditTeam(false);
              }}
              accessToken={accessToken}
              is_team_admin={is_team_admin(teams?.find((team) => team.team_id === selectedTeamId))}
              is_proxy_admin={userRole == "Admin"}
              userModels={userModels}
              editTeam={editTeam}
            />
          ) : (
            <TeamsHeaderTabs lastRefreshed={lastRefreshed} onRefresh={handleRefreshClick} userRole={userRole}>
              <TabPanel>
                <Text>
                  Click on &ldquo;Team ID&rdquo; to view team details <b>and</b> manage team members.
                </Text>
                <Grid numItems={1} className="gap-2 pt-2 pb-2 h-[75vh] w-full mt-2">
                  <Col numColSpan={1}>
                    <Card className="w-full mx-auto flex-auto overflow-hidden overflow-y-auto max-h-[50vh]">
                      <div className="border-b px-6 py-4">
                        <div className="flex flex-col space-y-4">
                          <TeamsFilters
                            filters={filters}
                            organizations={organizations}
                            showFilters={showFilters}
                            onToggleFilters={setShowFilters}
                            onChange={handleFilterChange}
                            onReset={handleFilterReset}
                          />
                        </div>
                      </div>
                      <TeamsTable
                        teams={teams}
                        currentOrg={currentOrg}
                        perTeamInfo={perTeamInfo}
                        userRole={userRole}
                        userId={userID}
                        setSelectedTeamId={setSelectedTeamId}
                        setEditTeam={setEditTeam}
                        onDeleteTeam={handleDelete}
                      />
                      {isDeleteModalOpen && (
                        <DeleteTeamModal
                          teams={teams}
                          teamToDelete={teamToDelete}
                          onCancel={cancelDelete}
                          onConfirm={confirmDelete}
                        />
                      )}
                    </Card>
                  </Col>
                </Grid>
              </TabPanel>
              <TabPanel>
                <AvailableTeamsPanel accessToken={accessToken} userID={userID} />
              </TabPanel>
              {isAdminRole(userRole || "") && (
                <TabPanel>
                  <TeamSSOSettings accessToken={accessToken} userID={userID || ""} userRole={userRole || ""} />
                </TabPanel>
              )}
            </TeamsHeaderTabs>
          )}
          {(userRole == "Admin" || userRole == "Org Admin") && (
            <CreateTeamModal
              isTeamModalVisible={isTeamModalVisible}
              handleOk={handleOk}
              handleCancel={handleCancel}
              currentOrg={currentOrg}
              organizations={organizations}
              teams={teams}
              setTeams={setTeams}
              modelAliases={modelAliases}
              setModelAliases={setModelAliases}
              loggingSettings={loggingSettings}
              setLoggingSettings={setLoggingSettings}
              setIsTeamModalVisible={setIsTeamModalVisible}
            />
          )}
        </Col>
      </Grid>
    </div>
  );
};

export default TeamsView;
