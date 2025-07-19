import React, { useState } from 'react';
import {
  Button as Button2,
  Modal,
  Typography,
  Divider,
  message,
  Table,
} from "antd";
import { Button } from '@tremor/react';
import { userBulkUpdateUserCall } from "./networking";
import { UserEditView } from "./user_edit_view";

const { Text, Title } = Typography;

interface BulkEditUserModalProps {
  visible: boolean;
  onCancel: () => void;
  selectedUsers: any[];
  possibleUIRoles: Record<string, Record<string, string>> | null;
  accessToken: string | null;
  onSuccess: () => void;
  teams: any[] | null;
  userRole: string | null;
  userModels: string[];
}

const BulkEditUserModal: React.FC<BulkEditUserModalProps> = ({
  visible,
  onCancel,
  selectedUsers,
  possibleUIRoles,
  accessToken,
  onSuccess,
  teams,
  userRole,
  userModels,
}) => {
  const [loading, setLoading] = useState(false);

  const handleCancel = () => {
    onCancel();
  };

  // Create a mock userData object for the UserEditView
  const mockUserData = {
    user_id: "bulk_edit",
    user_info: {
      user_email: "",
      user_role: "",
      teams: [],
      models: [],
      max_budget: null,
      spend: 0,
      metadata: {},
      created_at: null,
      updated_at: null,
    },
    keys: [],
    teams: teams || [],
  };

  const handleSubmit = async (formValues: any) => {
    if (!accessToken) {
      message.error("Access token not found");
      return;
    }

    setLoading(true);
    try {
      const userIds = selectedUsers.map(user => user.user_id);
      
      // Build the update payload - only include fields that have been changed from default/empty values
      const updatePayload: any = {};
      
      if (formValues.user_role && formValues.user_role !== "") {
        updatePayload.user_role = formValues.user_role;
      }
      
      if (formValues.max_budget !== null && formValues.max_budget !== undefined) {
        updatePayload.max_budget = formValues.max_budget;
      }

      if (formValues.models && formValues.models.length > 0) {
        updatePayload.models = formValues.models;
      }

      if (formValues.metadata && Object.keys(formValues.metadata).length > 0) {
        updatePayload.metadata = formValues.metadata;
      }

      // Check if any fields were actually updated
      if (Object.keys(updatePayload).length === 0) {
        message.error("Please modify at least one field to update");
        return;
      }

      await userBulkUpdateUserCall(accessToken, updatePayload, userIds);
      
      message.success(`Successfully updated ${userIds.length} user(s)`);
      onSuccess();
      onCancel();
    } catch (error) {
      console.error("Bulk update failed:", error);
      message.error("Failed to update users");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal
      visible={visible}
      onCancel={handleCancel}
      footer={null}
      title={`Bulk Edit ${selectedUsers.length} User(s)`}
      width={800}
    >
      <div className="mb-4">
        <Title level={5}>Selected Users ({selectedUsers.length}):</Title>
        <Table
          size="small"
          bordered
          dataSource={selectedUsers}
          pagination={false}
          scroll={{ y: 200 }}
          rowKey="user_id"
          columns={[
            {
              title: 'User ID',
              dataIndex: 'user_id',
              key: 'user_id',
              width: '30%',
              render: (text: string) => (
                <Text strong style={{ fontSize: '12px' }}>
                  {text.length > 20 ? `${text.slice(0, 20)}...` : text}
                </Text>
              ),
            },
            {
              title: 'Email',
              dataIndex: 'user_email',
              key: 'user_email',
              width: '25%',
              render: (text: string) => (
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  {text || 'No email'}
                </Text>
              ),
            },
            {
              title: 'Current Role',
              dataIndex: 'user_role',
              key: 'user_role',
              width: '25%',
              render: (role: string) => (
                <Text style={{ fontSize: '12px' }}>
                  {possibleUIRoles?.[role]?.ui_label || role}
                </Text>
              ),
            },
            {
              title: 'Budget',
              dataIndex: 'max_budget',
              key: 'max_budget',
              width: '20%',
              render: (budget: number | null) => (
                <Text style={{ fontSize: '12px' }}>
                  {budget !== null ? `$${budget}` : 'Unlimited'}
                </Text>
              ),
            },
          ]}
        />
      </div>

      <Divider />

      <div className="mb-4">
        <Text>
          <strong>Instructions:</strong> Fill in the fields below with the values you want to apply to all selected users. 
          You can bulk edit: role, budget, models, and metadata. Leave fields empty if you don't want to change them.
        </Text>
      </div>

      <UserEditView
        userData={mockUserData}
        onCancel={handleCancel}
        onSubmit={handleSubmit}
        teams={teams}
        accessToken={accessToken}
        userID="bulk_edit"
        userRole={userRole}
        userModels={userModels}
        possibleUIRoles={possibleUIRoles}
        isBulkEdit={true}
      />

      {loading && (
        <div style={{ textAlign: "center", marginTop: "10px" }}>
          <Text>Updating {selectedUsers.length} user(s)...</Text>
        </div>
      )}
    </Modal>
  );
};

export default BulkEditUserModal; 