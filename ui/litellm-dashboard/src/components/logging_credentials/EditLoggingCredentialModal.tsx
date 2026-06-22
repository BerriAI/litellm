import { Form, Modal } from "antd";
import React from "react";

import { CredentialAccess } from "../Settings/LoggingAndAlerts/LoggingCallbacks/types";
import NotificationsManager from "../molecules/notifications_manager";
import { credentialUpdateCall } from "../networking";
import AccessControlFields from "./AccessControlFields";

interface EditLoggingCredentialModalProps {
  accessToken: string;
  credentialName: string | null;
  access?: CredentialAccess;
  open: boolean;
  onClose: () => void;
  onSaved: () => void;
}

interface AccessForm {
  access?: CredentialAccess;
}

const EditLoggingCredentialModal: React.FC<EditLoggingCredentialModalProps> = ({
  accessToken,
  credentialName,
  access,
  open,
  onClose,
  onSaved,
}) => {
  // destroyOnClose remounts the Form each open, so initialValues re-seeds from the
  // current destination -- no effect syncing prop into state.
  const [form] = Form.useForm<AccessForm>();

  const handleSave = async () => {
    if (!credentialName) return;
    const current = form.getFieldsValue().access ?? {};
    // Always send the full access object: credential_info merges server-side, so a
    // sparse patch could never clear a bucket. A global grant supersedes team/org.
    const next: CredentialAccess = current.global
      ? { global: true, teams: [], orgs: [] }
      : { global: false, teams: current.teams ?? [], orgs: current.orgs ?? [] };
    try {
      await credentialUpdateCall(accessToken, credentialName, {
        credential_name: credentialName,
        credential_values: {},
        credential_info: { access: next },
      });
      NotificationsManager.success("Access updated");
      onSaved();
      onClose();
    } catch (error) {
      NotificationsManager.fromBackend(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <Modal
      title={`Edit scope${credentialName ? ` — ${credentialName}` : ""}`}
      open={open}
      onCancel={onClose}
      onOk={handleSave}
      okText="Save"
      destroyOnClose
    >
      <Form<AccessForm> form={form} layout="vertical" preserve={false} initialValues={{ access: access ?? {} }}>
        <Form.Item name="access" noStyle>
          <AccessControlFields />
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default EditLoggingCredentialModal;
