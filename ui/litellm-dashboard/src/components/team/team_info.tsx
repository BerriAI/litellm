import React, { useState, useEffect } from "react";
import {
  Card,
  Title,
  Text,
  Tab,
  TabList,
  TabGroup,
  TabPanel,
  TabPanels,
  Grid,
  Badge,
} from "@tremor/react";
import { teamInfoCall } from "@/components/networking";
import TeamMemberModal from "@/components/team/edit_membership";
import { Button, message } from "antd";

interface TeamMember {
  user_id: string;
  user_email: string | null;
  role: string;
}

interface TeamData {
  team_id: string;
  team_info: {
    team_alias: string;
    team_id: string;
    organization_id: string | null;
    admins: string[];
    members: string[];
    members_with_roles: TeamMember[];
    metadata: Record<string, any>;
    tpm_limit: number | null;
    rpm_limit: number | null;
    max_budget: number | null;
    budget_duration: string | null;
    models: string[];
    blocked: boolean;
    spend: number;
    max_parallel_requests: number | null;
    budget_reset_at: string | null;
    model_id: string | null;
    litellm_model_table: string | null;
    created_at: string;
  };
  keys: any[];
  team_memberships: any[];
}

interface TeamInfoProps {
  teamId: string;
  onClose: () => void;
  accessToken: string | null;
}

const TeamInfoView: React.FC<TeamInfoProps> = ({ teamId, onClose, accessToken }) => {
  const [teamData, setTeamData] = useState<TeamData | null>(null);
  const [loading, setLoading] = useState(true);
  const [isMemberModalOpen, setIsMemberModalOpen] = useState(false);

  useEffect(() => {
    const fetchTeamInfo = async () => {
      try {
        setLoading(true);
        if (!accessToken) {
          return;
        }
        const response = await teamInfoCall(accessToken, teamId);
        setTeamData(response);
      } catch (error) {
        message.error("Failed to load team information");
        console.error("Error fetching team info:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchTeamInfo();
  }, [teamId]);

  if (loading) {
    return <div className="p-4">Loading...</div>;
  }

  if (!teamData?.team_info) {
    return <div className="p-4">Team not found</div>;
  }

  const { team_info: info } = teamData;

  return (
    <div className="p-4">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button onClick={onClose} className="mb-4">← Back</Button>
          <Title>{info.team_alias}</Title>
          <Text className="text-gray-500 font-mono">{info.team_id}</Text>
        </div>
        <Button type="primary" onClick={() => setIsMemberModalOpen(true)}>
          Manage Members
        </Button>
      </div>

      <TabGroup>
        <TabList className="mb-4">
          <Tab>Overview</Tab>
          <Tab>Members</Tab>
          <Tab>Settings</Tab>
        </TabList>

        <TabPanels>
          <TabPanel>
            <Grid numItems={1} numItemsSm={2} numItemsLg={3} className="gap-6">
              <Card>
                <Text>Budget Status</Text>
                <div className="mt-2">
                  <Title>${info.spend.toFixed(6)}</Title>
                  <Text>of {info.max_budget === null ? "Unlimited" : `$${info.max_budget}`}</Text>
                  {info.budget_duration && (
                    <Text className="text-gray-500">Reset: {info.budget_duration}</Text>
                  )}
                </div>
              </Card>

              <Card>
                <Text>Rate Limits</Text>
                <div className="mt-2">
                  <Text>TPM: {info.tpm_limit || 'Unlimited'}</Text>
                  <Text>RPM: {info.rpm_limit || 'Unlimited'}</Text>
                  {info.max_parallel_requests && (
                    <Text>Max Parallel Requests: {info.max_parallel_requests}</Text>
                  )}
                </div>
              </Card>

              <Card>
                <Text>Models</Text>
                <div className="mt-2 flex flex-wrap gap-2">
                  {info.models.map((model, index) => (
                    <Badge key={index} color="red">
                      {model}
                    </Badge>
                  ))}
                </div>
              </Card>
            </Grid>
          </TabPanel>

          <TabPanel>
            <Card>
              <Title>Team Members ({info.members_with_roles.length})</Title>
              <div className="mt-4 space-y-4">
                {info.members_with_roles.map((member, index) => (
                  <div key={index} className="flex justify-between items-center p-4 bg-gray-50 rounded-lg">
                    <div>
                      <Text className="font-medium">{member.user_id}</Text>
                      {member.user_email && (
                        <Text className="text-gray-500">{member.user_email}</Text>
                      )}
                    </div>
                    <Badge color={member.role === 'admin' ? 'red' : 'blue'}>
                      {member.role}
                    </Badge>
                  </div>
                ))}
              </div>
            </Card>
          </TabPanel>

          <TabPanel>
            <Card>
              <Title>Team Settings</Title>
              <div className="mt-4 space-y-4">
                <div>
                  <Text className="font-medium">Team ID</Text>
                  <Text className="font-mono">{info.team_id}</Text>
                </div>
                <div>
                  <Text className="font-medium">Created At</Text>
                  <Text>{new Date(info.created_at).toLocaleString()}</Text>
                </div>
                <div>
                  <Text className="font-medium">Status</Text>
                  <Badge color={info.blocked ? 'red' : 'green'}>
                    {info.blocked ? 'Blocked' : 'Active'}
                  </Badge>
                </div>
              </div>
            </Card>
          </TabPanel>
        </TabPanels>
      </TabGroup>

      <TeamMemberModal
        open={isMemberModalOpen}
        onClose={() => setIsMemberModalOpen(false)}
        teamId={teamId}
      />
    </div>
  );
};

export default TeamInfoView;