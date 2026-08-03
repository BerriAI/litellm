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
  // The destination's stored credential_info. PATCH replaces credential_info
  // wholesale, so the whole object is resent with only access swapped; sending
  // access alone would drop credential_type/description and stop the row being
  // a logging destination at all.
  credentialInfo?: Record<string, unknown>;
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
  credentialInfo,
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
    // Always send the full access object; a global grant supersedes team/org.
    const next: CredentialAccess = current.global
      ? { global: true, teams: [], orgs: [] }
      : { global: false, teams: current.teams ?? [], orgs: current.orgs ?? [] };
    try {
      await credentialUpdateCall(accessToken, credentialName, {
        credential_name: credentialName,
        credential_values: {},
        credential_info: { ...(credentialInfo ?? {}), access: next },
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
