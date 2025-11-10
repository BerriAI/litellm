import React from "react";
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
} from "@tremor/react";
import { InfoCircleOutlined } from "@ant-design/icons";
import { Tooltip } from "antd";
import { TeamData } from "./team_info";
import { PencilAltIcon, TrashIcon } from "@heroicons/react/outline";
import { formatNumberWithCommas } from "@/utils/dataUtils";

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
  // Helper function to convert scientific notation to normal decimal format
  const formatNumber = (value: number | null): string => {
    if (value === null || value === undefined) return "0";

    if (typeof value === "number") {
      // Convert scientific notation to normal decimal
      const normalNumber = Number(value);

      // If it's a whole number, return it without decimals
      if (normalNumber === Math.floor(normalNumber)) {
        return normalNumber.toString();
      }

      // For decimal numbers, use toFixed and remove trailing zeros
      return formatNumberWithCommas(normalNumber, 8).replace(/\.?0+$/, "");
    }

    return "0";
  };

  // Helper function to get spend for a user
  const getUserSpend = (userId: string | null): number | null => {
    if (!userId) return 0;
    const membership = teamData.team_memberships.find((tm) => tm.user_id === userId);
    return membership?.spend || 0;
  };

  const getUserBudget = (userId: string | null): string | null => {
    if (!userId) return null;
    const membership = teamData.team_memberships.find((tm) => tm.user_id === userId);
    console.log(`membership=${membership}`);
    const maxBudget = membership?.litellm_budget_table?.max_budget;
    if (maxBudget === null || maxBudget === undefined) {
      return null;
    }
    return formatNumber(maxBudget);
  };

  // Helper function to get rate limits for a user
  const getUserRateLimits = (userId: string | null): string => {
    if (!userId) return "No Limits";
    const membership = teamData.team_memberships.find((tm) => tm.user_id === userId);
    const rpmLimit = membership?.litellm_budget_table?.rpm_limit;
    const tpmLimit = membership?.litellm_budget_table?.tpm_limit;

    const rpmText = rpmLimit ? `${formatNumber(rpmLimit)} RPM` : null;
    const tpmText = tpmLimit ? `${formatNumber(tpmLimit)} TPM` : null;

    const limits = [rpmText, tpmText].filter(Boolean);
    return limits.length > 0 ? limits.join(" / ") : "No Limits";
  };

  return (
    <div className="space-y-4">
      <Card className="w-full mx-auto flex-auto overflow-auto max-h-[50vh]">
        <div className="overflow-x-auto">
          <Table className="min-w-full">
            <TableHead>
              <TableRow>
                <TableHeaderCell>User ID</TableHeaderCell>
                <TableHeaderCell>User Email</TableHeaderCell>
                <TableHeaderCell>Role</TableHeaderCell>
                <TableHeaderCell>
                  Team Member Spend (USD){" "}
                  <Tooltip title="This is the amount spent by a user in the team.">
                    <InfoCircleOutlined />
                  </Tooltip>
                </TableHeaderCell>
                <TableHeaderCell>Team Member Budget (USD)</TableHeaderCell>
                <TableHeaderCell>
                  Team Member Rate Limits{" "}
                  <Tooltip title="Rate limits for this member's usage within this team.">
                    <InfoCircleOutlined />
                  </Tooltip>
                </TableHeaderCell>
                <TableHeaderCell className="sticky right-0 bg-white z-10 border-l border-gray-200">
                  Actions
                </TableHeaderCell>
              </TableRow>
            </TableHead>

            <TableBody>
              {teamData.team_info.members_with_roles.map((member: Member, index: number) => (
                <TableRow key={index}>
                  <TableCell>
                    <Text className="font-mono">{member.user_id}</Text>
                  </TableCell>
                  <TableCell>
                    <Text className="font-mono">{member.user_email ? member.user_email : "No Email"}</Text>
                  </TableCell>
                  <TableCell>
                    <Text className="font-mono">{member.role}</Text>
                  </TableCell>
                  <TableCell>
                    <Text className="font-mono">${formatNumberWithCommas(getUserSpend(member.user_id), 4)}</Text>
                  </TableCell>
                  <TableCell>
                    <Text className="font-mono">
                      {getUserBudget(member.user_id)
                        ? `$${formatNumberWithCommas(Number(getUserBudget(member.user_id)), 4)}`
                        : "No Limit"}
                    </Text>
                  </TableCell>
                  <TableCell>
                    <Text className="font-mono">{getUserRateLimits(member.user_id)}</Text>
                  </TableCell>
                  <TableCell className="sticky right-0 bg-white z-10 border-l border-gray-200">
                    {canEditTeam && (
                      <div className="flex gap-2">
                        <Icon
                          icon={PencilAltIcon}
                          size="sm"
                          onClick={() => {
                            // Get budget and rate limit data from team membership
                            const membership = teamData.team_memberships.find((tm) => tm.user_id === member.user_id);
                            const enhancedMember = {
                              ...member,
                              max_budget_in_team: membership?.litellm_budget_table?.max_budget || null,
                              tpm_limit: membership?.litellm_budget_table?.tpm_limit || null,
                              rpm_limit: membership?.litellm_budget_table?.rpm_limit || null,
                            };
                            setSelectedEditMember(enhancedMember);
                            setIsEditMemberModalVisible(true);
                          }}
                          className="cursor-pointer hover:text-blue-600"
                        />
                        <Icon
                          icon={TrashIcon}
                          size="sm"
                          onClick={() => handleMemberDelete(member)}
                          className="cursor-pointer hover:text-red-600"
                        />
                      </div>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </Card>
      <TremorButton onClick={() => setIsAddMemberModalVisible(true)}>Add Member</TremorButton>
    </div>
  );
};

export default TeamMembersComponent;
