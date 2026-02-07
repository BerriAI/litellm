import React from "react";
import { Modal, Typography } from "antd";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { Text, Button } from "@tremor/react";
import NotificationsManager from "./molecules/notifications_manager";

export interface InvitationLink {
  id: string;
  user_id: string;
  is_accepted: boolean;
  accepted_at: Date | null;
  expires_at: Date;
  created_at: Date;
  created_by: string;
  updated_at: Date;
  updated_by: string;
  has_user_setup_sso: boolean;
}

interface OnboardingProps {
  isInvitationLinkModalVisible: boolean;
  setIsInvitationLinkModalVisible: React.Dispatch<React.SetStateAction<boolean>>;
  baseUrl: string;
  invitationLinkData: InvitationLink | null;
  modalType?: "invitation" | "resetPassword";
}

export default function OnboardingModal({
  isInvitationLinkModalVisible,
  setIsInvitationLinkModalVisible,
  baseUrl,
  invitationLinkData,
  modalType = "invitation",
}: OnboardingProps) {
  const { Title, Paragraph } = Typography;
  const handleInvitationOk = () => {
    setIsInvitationLinkModalVisible(false);
  };

  const handleInvitationCancel = () => {
    setIsInvitationLinkModalVisible(false);
  };

  const getInvitationUrl = () => {
    if (!baseUrl) {
      return "";
    }
    const baseUrlObj = new URL(baseUrl);
    const basePath = baseUrlObj.pathname; // This will be "/litellm" or ""
    const path = basePath && basePath !== "/" ? `${basePath}/ui` : "ui";
    // Get the path from the base URL
    if (invitationLinkData?.has_user_setup_sso) {
      return new URL(path, baseUrl).toString();
    }
    let urlPath = `${path}?invitation_id=${invitationLinkData?.id}`;
    if (modalType === "resetPassword") {
      urlPath += "&action=reset_password";
    }
    const url = new URL(urlPath, baseUrl).toString();
    return url;
  };

  return (
    <Modal
      title={modalType === "invitation" ? "Invitation Link" : "Reset Password Link"}
      open={isInvitationLinkModalVisible}
      width={800}
      footer={null}
      onOk={handleInvitationOk}
      onCancel={handleInvitationCancel}
    >
      <Paragraph>
        {modalType === "invitation"
          ? "Copy and send the generated link to onboard this user to the proxy."
          : "Copy and send the generated link to the user to reset their password."}
      </Paragraph>
      <div className="flex justify-between pt-5 pb-2">
        <Text className="text-base">User ID</Text>
        <Text>{invitationLinkData?.user_id}</Text>
      </div>
      <div className="flex justify-between pt-5 pb-2">
        <Text>{modalType === "invitation" ? "Invitation Link" : "Reset Password Link"}</Text>
        <Text>
          <Text>{getInvitationUrl()}</Text>
        </Text>
      </div>
      <div className="flex justify-end mt-5">
        <CopyToClipboard text={getInvitationUrl()} onCopy={() => NotificationsManager.success("Copied!")}>
          <Button variant="primary">
            {modalType === "invitation" ? "Copy invitation link" : "Copy password reset link"}
          </Button>
        </CopyToClipboard>
      </div>
    </Modal>
  );
}
