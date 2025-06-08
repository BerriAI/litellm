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
  return (
    <div className="space-y-4">
      <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh]">
        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>User ID</TableHeaderCell>
              <TableHeaderCell>User Email</TableHeaderCell>
              <TableHeaderCell>Role</TableHeaderCell>
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
