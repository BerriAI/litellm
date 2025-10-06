"use client";

import React, { useState, useEffect } from "react";
import { Button } from "@tremor/react";
import { Modal, Form } from "antd";
import { getPossibleUserRoles, Organization } from "@/components/networking";
import { rolesWithWriteAccess } from "@/utils/roles";
import { Team } from "@/components/key_team_helpers/key_list";
import CreateKeyModal from "@/app/dashboard/virtual-keys/components/CreateKeyModal/CreateKeyModal";
import CreateUserModal from "@/app/dashboard/components/modals/CreateUserModal";
import useAuthorized from "@/app/dashboard/hooks/useAuthorized";

interface CreateKeyProps {
  team: Team | null;
  userRole: string | null;
  data: any[] | null;
  addKey: (data: any) => void;
}

interface User {
  user_id: string;
  user_email: string;
  role?: string;
}

interface UserOption {
  label: string;
  value: string;
  user: User;
}
const CreateKey: React.FC<CreateKeyProps> = ({ team, userRole, data, addKey }) => {
  const { accessToken } = useAuthorized();
  const [form] = Form.useForm();
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [isCreateUserModalVisible, setIsCreateUserModalVisible] = useState(false);
  const [newlyCreatedUserId, setNewlyCreatedUserId] = useState<string | null>(null);
  const [possibleUIRoles, setPossibleUIRoles] = useState<Record<string, Record<string, string>>>({});

  useEffect(() => {
    const fetchPossibleRoles = async () => {
      try {
        if (accessToken) {
          // Check if roles are cached in session storage
          const cachedRoles = sessionStorage.getItem("possibleUserRoles");
          if (cachedRoles) {
            setPossibleUIRoles(JSON.parse(cachedRoles));
          } else {
            const availableUserRoles = await getPossibleUserRoles(accessToken);
            sessionStorage.setItem("possibleUserRoles", JSON.stringify(availableUserRoles));
            setPossibleUIRoles(availableUserRoles);
          }
        }
      } catch (error) {
        console.error("Error fetching possible user roles:", error);
      }
    };

    fetchPossibleRoles();
  }, [accessToken]);

  // Add a callback function to handle user creation
  const handleUserCreated = (userId: string) => {
    setNewlyCreatedUserId(userId);
    form.setFieldsValue({ user_id: userId });
    setIsCreateUserModalVisible(false);
  };

  const handleUserSelect = (_value: string, option: UserOption): void => {
    const selectedUser = option.user;
    form.setFieldsValue({
      user_id: selectedUser.user_id,
    });
  };

  return (
    <div>
      {userRole && rolesWithWriteAccess.includes(userRole) && (
        <Button className="mx-auto" onClick={() => setIsModalVisible(true)}>
          + Create New Key
        </Button>
      )}
      <CreateKeyModal
        isModalVisible={isModalVisible}
        form={form}
        handleUserSelect={handleUserSelect}
        setIsCreateUserModalVisible={setIsCreateUserModalVisible}
        team={team}
        setIsModalVisible={setIsModalVisible}
        data={data}
        addKey={addKey}
      />

      {isCreateUserModalVisible && (
        <Modal
          title="Create New User"
          open={isCreateUserModalVisible}
          onCancel={() => setIsCreateUserModalVisible(false)}
          footer={null}
          width={800}
        >
          <CreateUserModal possibleUIRoles={possibleUIRoles} onUserCreated={handleUserCreated} isEmbedded={true} />
        </Modal>
      )}
    </div>
  );
};

export default CreateKey;
