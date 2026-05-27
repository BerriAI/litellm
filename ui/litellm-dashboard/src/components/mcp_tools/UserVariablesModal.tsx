import React from "react";
import { Modal, Tag, Typography } from "antd";
import { MCPUserVariablesGlobalStatus } from "./types";
import UserVariablesForm from "./UserVariablesForm";

const { Text, Title } = Typography;

interface UserVariablesModalProps {
  open: boolean;
  accessToken: string | null;
  onClose: () => void;
  onSaved?: (status: MCPUserVariablesGlobalStatus) => void;
}

/**
 * User-facing modal for filling in per-user MCP variables.
 *
 * Wraps the reusable {@link UserVariablesForm} (which talks to the GLOBAL,
 * per-user endpoints ``/v1/mcp/user/variables``). The form is mounted only
 * while the modal is open (``destroyOnHidden``), so it re-fetches the latest
 * status each time the modal is opened from a card's "Set" button or a
 * deep-link. Values are write-only and never prefilled.
 */
const UserVariablesModal: React.FC<UserVariablesModalProps> = ({
  open,
  accessToken,
  onClose,
  onSaved,
}) => {
  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      width={520}
      destroyOnHidden
      title={
        <div>
          <div className="flex items-center gap-2">
            <Title level={5} style={{ margin: 0 }}>
              Set your credentials
            </Title>
            <Tag color="blue">Per-user</Tag>
          </div>
          <Text type="secondary" className="text-xs">
            These values apply to every MCP server that requires them.
          </Text>
        </div>
      }
    >
      <div className="mt-2">
        <UserVariablesForm
          accessToken={accessToken}
          onSaved={(saved) => {
            if (onSaved) onSaved(saved);
            onClose();
          }}
        />
      </div>
    </Modal>
  );
};

export default UserVariablesModal;
