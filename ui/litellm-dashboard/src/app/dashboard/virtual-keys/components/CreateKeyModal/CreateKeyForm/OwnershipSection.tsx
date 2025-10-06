"use client";

import { Button as Button2, Form, Radio, Select, Tooltip } from "antd";
import { Title } from "@tremor/react";
import { InfoCircleOutlined } from "@ant-design/icons";
import TeamDropdown from "@/components/common_components/team_dropdown";
import React from "react";
import { Team } from "@/components/key_team_helpers/key_list";
import { UserOption } from "@/app/dashboard/virtual-keys/components/CreateKeyModal/types";

interface OwnershipSectionProps {
  team: Team | null;
  teams: Team[] | null;
  userRole: string | null;
  userOptions: UserOption[];
  keyOwner: string;
  setKeyOwner: React.Dispatch<React.SetStateAction<string>>;
  handleUserSearch: (q: string) => void;
  userSearchLoading: boolean;
  handleUserSelect: (q: string, option: UserOption) => void;
  setSelectedCreateKeyTeam: React.Dispatch<React.SetStateAction<Team | null>>;
  setIsCreateUserModalVisible: (visible: boolean) => void;
}

const OwnershipSection = ({
  team,
  teams,
  userRole,
  userOptions,
  keyOwner,
  setKeyOwner,
  handleUserSearch,
  userSearchLoading,
  handleUserSelect,
  setSelectedCreateKeyTeam,
  setIsCreateUserModalVisible,
}: OwnershipSectionProps) => {
  return (
    <div className="mb-8">
      <Title className="mb-4">Key Ownership</Title>
      <Form.Item
        label={
          <span>
            Owned By{" "}
            <Tooltip title="Select who will own this API key">
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
        className="mb-4"
      >
        <Radio.Group onChange={(e) => setKeyOwner(e.target.value)} value={keyOwner}>
          <Radio value="you">You</Radio>
          <Radio value="service_account">Service Account</Radio>
          {userRole === "Admin" && <Radio value="another_user">Another User</Radio>}
        </Radio.Group>
      </Form.Item>

      {keyOwner === "another_user" && (
        <Form.Item
          label={
            <span>
              User ID{" "}
              <Tooltip title="The user who will own this key and be responsible for its usage">
                <InfoCircleOutlined style={{ marginLeft: "4px" }} />
              </Tooltip>
            </span>
          }
          name="user_id"
          className="mt-4"
          rules={[
            {
              required: keyOwner === "another_user",
              message: `Please input the user ID of the user you are assigning the key to`,
            },
          ]}
        >
          <div>
            <div style={{ display: "flex", marginBottom: "8px" }}>
              <Select
                showSearch
                placeholder="Type email to search for users"
                filterOption={false}
                onSearch={handleUserSearch}
                onSelect={(value, option) => handleUserSelect(value, option as UserOption)}
                options={userOptions}
                loading={userSearchLoading}
                allowClear
                style={{ width: "100%" }}
                notFoundContent={userSearchLoading ? "Searching..." : "No users found"}
              />
              <Button2 onClick={() => setIsCreateUserModalVisible(true)} style={{ marginLeft: "8px" }}>
                Create User
              </Button2>
            </div>
            <div className="text-xs text-gray-500">Search by email to find users</div>
          </div>
        </Form.Item>
      )}
      <Form.Item
        label={
          <span>
            Team{" "}
            <Tooltip title="The team this key belongs to, which determines available models and budget limits">
              <InfoCircleOutlined style={{ marginLeft: "4px" }} />
            </Tooltip>
          </span>
        }
        name="team_id"
        initialValue={team ? team.team_id : null}
        className="mt-4"
        rules={[
          {
            required: keyOwner === "service_account",
            message: "Please select a team for the service account",
          },
        ]}
        help={keyOwner === "service_account" ? "required" : ""}
      >
        <TeamDropdown
          teams={teams}
          onChange={(teamId) => {
            const selectedTeam = teams?.find((t) => t.team_id === teamId) || null;
            setSelectedCreateKeyTeam(selectedTeam);
          }}
        />
      </Form.Item>
    </div>
  );
};

export default OwnershipSection;
