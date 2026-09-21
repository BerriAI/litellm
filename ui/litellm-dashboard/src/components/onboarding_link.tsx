import React from "react";
import { Button } from "@/components/ui/button";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { toast } from "@/lib/toast";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

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

export function buildOnboardingUrl({
  baseUrl,
  invitationId,
  hasUserSetupSso,
  resetPassword,
}: {
  baseUrl: string;
  invitationId: string | undefined;
  hasUserSetupSso: boolean;
  resetPassword: boolean;
}): string {
  if (!baseUrl) {
    return "";
  }
  const basePath = new URL(baseUrl).pathname;
  const uiPath = basePath && basePath !== "/" ? `${basePath}/ui` : "ui";
  if (hasUserSetupSso) {
    return new URL(uiPath, baseUrl).toString();
  }
  if (!invitationId) {
    return "";
  }
  const action = resetPassword ? "&action=reset_password" : "";
  return new URL(`${uiPath}/onboarding?invitation_id=${invitationId}${action}`, baseUrl).toString();
}

export default function OnboardingModal({
  isInvitationLinkModalVisible,
  setIsInvitationLinkModalVisible,
  baseUrl,
  invitationLinkData,
  modalType = "invitation",
}: OnboardingProps) {
  const handleInvitationCancel = () => {
    setIsInvitationLinkModalVisible(false);
  };

  const getInvitationUrl = () =>
    buildOnboardingUrl({
      baseUrl,
      invitationId: invitationLinkData?.id,
      hasUserSetupSso: invitationLinkData?.has_user_setup_sso ?? false,
      resetPassword: modalType === "resetPassword",
    });

  return (
    <Dialog open={isInvitationLinkModalVisible} onOpenChange={(open) => !open && handleInvitationCancel()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[800px]">
        <DialogHeader>
          <DialogTitle>{modalType === "invitation" ? "Invitation Link" : "Reset Password Link"}</DialogTitle>
        </DialogHeader>
        <p className="text-sm text-foreground">
          {modalType === "invitation"
            ? "Copy and send the generated link to onboard this user to the proxy."
            : "Copy and send the generated link to the user to reset their password."}
        </p>
        <div className="flex justify-between pt-5 pb-2">
          <p className="text-base">User ID</p>
          <p className="text-sm">{invitationLinkData?.user_id}</p>
        </div>
        <div className="flex justify-between pt-5 pb-2">
          <p className="text-sm">{modalType === "invitation" ? "Invitation Link" : "Reset Password Link"}</p>
          <p className="text-sm">{getInvitationUrl()}</p>
        </div>
        <div className="flex justify-end mt-5">
          <CopyToClipboard text={getInvitationUrl()} onCopy={() => toast.success("Copied!")}>
            <Button>{modalType === "invitation" ? "Copy invitation link" : "Copy password reset link"}</Button>
          </CopyToClipboard>
        </div>
      </DialogContent>
    </Dialog>
  );
}
