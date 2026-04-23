import React from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { CopyToClipboard } from "react-copy-to-clipboard";
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
  setIsInvitationLinkModalVisible: React.Dispatch<
    React.SetStateAction<boolean>
  >;
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
  const handleClose = () => setIsInvitationLinkModalVisible(false);

  const getInvitationUrl = () => {
    if (!baseUrl) return "";
    const baseUrlObj = new URL(baseUrl);
    const basePath = baseUrlObj.pathname;
    const path = basePath && basePath !== "/" ? `${basePath}/ui` : "ui";
    if (invitationLinkData?.has_user_setup_sso) {
      return new URL(path, baseUrl).toString();
    }
    let urlPath = `${path}?invitation_id=${invitationLinkData?.id}`;
    if (modalType === "resetPassword") {
      urlPath += "&action=reset_password";
    }
    return new URL(urlPath, baseUrl).toString();
  };

  return (
    <Dialog
      open={isInvitationLinkModalVisible}
      onOpenChange={(o) => (!o ? handleClose() : undefined)}
    >
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle>
            {modalType === "invitation"
              ? "Invitation Link"
              : "Reset Password Link"}
          </DialogTitle>
          <DialogDescription>
            {modalType === "invitation"
              ? "Copy and send the generated link to onboard this user to the proxy."
              : "Copy and send the generated link to the user to reset their password."}
          </DialogDescription>
        </DialogHeader>
        <div className="flex justify-between pt-2 pb-2">
          <span className="text-base">User ID</span>
          <span>{invitationLinkData?.user_id}</span>
        </div>
        <div className="flex justify-between pt-2 pb-2 gap-4">
          <span>
            {modalType === "invitation"
              ? "Invitation Link"
              : "Reset Password Link"}
          </span>
          <span className="text-right break-all font-mono text-sm">
            {getInvitationUrl()}
          </span>
        </div>
        <div className="flex justify-end mt-3">
          <CopyToClipboard
            text={getInvitationUrl()}
            onCopy={() => NotificationsManager.success("Copied!")}
          >
            <Button>
              {modalType === "invitation"
                ? "Copy invitation link"
                : "Copy password reset link"}
            </Button>
          </CopyToClipboard>
        </div>
      </DialogContent>
    </Dialog>
  );
}
