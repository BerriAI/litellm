import React from 'react';
import { Member } from "@/components/networking";
import {
  Card,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
  Text,
  Icon,
  Button as TremorButton,
} from '@tremor/react';
import {
    TeamData,
} from './team_info';
import { PencilAltIcon, PlusIcon, TrashIcon } from "@heroicons/react/outline";

interface TeamMembersComponentProps {
  teamData: TeamData;
  canEditTeam: boolean;
  handleMemberDelete: (member: Member) => void;
  setSelectedEditMember: (member: Member) => void;
  setIsEditMemberModalVisible: (visible: boolean) => void;
  setIsAddMemberModalVisible: (visible: boolean) => void;
}

const TeamMembersComponent: React.FC<TeamMembersComponentProps> = ({
  teamData,
  canEditTeam,
  handleMemberDelete,
  setSelectedEditMember,
  setIsEditMemberModalVisible,
  setIsAddMemberModalVisible,
}) => {
  // Helper function to get spend for a user
  const getUserSpend = (userId: string | null): number | null => {
    if (!userId) return 0;
    const membership = teamData.team_memberships.find(tm => tm.user_id === userId);
    return membership?.spend || 0;
  };

  const getUserBudget = (userId: string | null): number | null => {
    if (!userId) return null;
    const membership = teamData.team_memberships.find(tm => tm.user_id === userId);
    console.log(`membership=${membership}`);
    return membership?.litellm_budget_table?.max_budget || null;
  };

  return (
    <div className="space-y-4">
      <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh]">
        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>User ID</TableHeaderCell>
              <TableHeaderCell>User Email</TableHeaderCell>
              <TableHeaderCell>Role</TableHeaderCell>
              <TableHeaderCell>Spend (USD)</TableHeaderCell>
              <TableHeaderCell>Team Member Budget (USD)</TableHeaderCell>
              <TableHeaderCell></TableHeaderCell>
            </TableRow>
          </TableHead>

          <TableBody>
            {teamData.team_info.members_with_roles.map((member: Member, index: number) => (
              <TableRow key={index}>
                <TableCell>
                  <Text className="font-mono">{member.user_id}</Text>
                </TableCell>
                <TableCell>
                  <Text className="font-mono">{member.user_email}</Text>
                </TableCell>
                <TableCell>
                  <Text className="font-mono">{member.role}</Text>
                </TableCell>
                <TableCell>
                  <Text className="font-mono">{getUserSpend(member.user_id) ? `$${getUserSpend(member.user_id)}` : '$0'}</Text>
                </TableCell>
                <TableCell>

                  <Text className="font-mono">{getUserBudget(member.user_id) ? `$${getUserBudget(member.user_id)}` : 'No Limit'}</Text>
                </TableCell>
                <TableCell>
                  {canEditTeam && (
                    <>
                      <Icon
                        icon={PencilAltIcon}
                        size="sm"
                        onClick={() => {
                          setSelectedEditMember(member);
                          setIsEditMemberModalVisible(true);
                        }}
                      />
                      <Icon
                        icon={TrashIcon}
                        size="sm"
                        onClick={() => handleMemberDelete(member)}
                      />
                    </>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>
      <TremorButton onClick={() => setIsAddMemberModalVisible(true)}>
        Add Member
      </TremorButton>
    </div>
  );
};

export default TeamMembersComponent;
