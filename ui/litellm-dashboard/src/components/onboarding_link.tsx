import React, { useState, useEffect } from "react";
import {
  Button as Button2,
  Modal,
  Form,
  Input,
  Select as Select2,
  InputNumber,
  message,
  Typography,
} from "antd";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { Text, Button } from "@tremor/react";
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
  setIsInvitationLinkModalVisible: React.Dispatch<
    React.SetStateAction<boolean>
  >;
  baseUrl: string;
  invitationLinkData: InvitationLink | null;
}

const OnboardingModal: React.FC<OnboardingProps> = ({
  isInvitationLinkModalVisible,
  setIsInvitationLinkModalVisible,
  baseUrl,
  invitationLinkData,
}) => {
  const { Title, Paragraph } = Typography;
  const handleInvitationOk = () => {
    setIsInvitationLinkModalVisible(false);
  };

  const handleInvitationCancel = () => {
    setIsInvitationLinkModalVisible(false);
  };

  const getInvitationUrl = () => {
    if (invitationLinkData?.has_user_setup_sso) {
      return `${baseUrl}/ui`;
    }
    return `${baseUrl}/ui?invitation_id=${invitationLinkData?.id}`;
  };

  return (
    <Modal
      title="Invitation Link"
      visible={isInvitationLinkModalVisible}
      width={800}
      footer={null}
      onOk={handleInvitationOk}
      onCancel={handleInvitationCancel}
    >
      {/* {JSON.stringify(invitationLinkData)} */}
      <Paragraph>
        Copy and send the generated link to onboard this user to the proxy.
      </Paragraph>
      <div className="flex justify-between pt-5 pb-2">
        <Text className="text-base">User ID</Text>
        <Text>{invitationLinkData?.user_id}</Text>
      </div>
      <div className="flex justify-between pt-5 pb-2">
        <Text>Invitation Link</Text>
        <Text>
        <Text>{getInvitationUrl()}</Text>
        </Text>
      </div>
      <div className="flex justify-end mt-5">
        <div></div>
        <CopyToClipboard
          text={getInvitationUrl()}
          onCopy={() => message.success("Copied!")}
        >
          <Button variant="primary">Copy invitation link</Button>
        </CopyToClipboard>
      </div>
    </Modal>
  );
};

export default OnboardingModal;
