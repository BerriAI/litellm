import React, { useId, useRef } from "react";
import { Copy } from "lucide-react";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { Label } from "@/components/ui/label";
import { copyToClipboard } from "@/utils/dataUtils";

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
  const linkFieldId = useId();
  const copyButtonRef = useRef<HTMLButtonElement>(null);
  const isInvitation = modalType === "invitation";
  const invitationUrl = buildOnboardingUrl({
    baseUrl,
    invitationId: invitationLinkData?.id,
    hasUserSetupSso: invitationLinkData?.has_user_setup_sso ?? false,
    resetPassword: !isInvitation,
  });

  return (
    <Dialog
      open={isInvitationLinkModalVisible}
      onOpenChange={(open) => !open && setIsInvitationLinkModalVisible(false)}
    >
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-xl" initialFocus={copyButtonRef}>
        <DialogHeader>
          <DialogTitle>{isInvitation ? "Invitation Link" : "Reset Password Link"}</DialogTitle>
          <DialogDescription>
            {isInvitation
              ? "Copy and send the generated link to onboard this user to the proxy."
              : "Copy and send the generated link to the user to reset their password."}
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <p className="text-xs text-muted-foreground">User ID</p>
            <p className="font-mono text-sm break-all">{invitationLinkData?.user_id}</p>
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor={linkFieldId} className="text-xs font-normal text-muted-foreground">
              {isInvitation ? "Invitation link" : "Reset password link"}
            </Label>
            <InputGroup>
              <InputGroupInput id={linkFieldId} readOnly value={invitationUrl} className="font-mono text-xs" />
              <InputGroupAddon align="inline-end">
                <InputGroupButton
                  ref={copyButtonRef}
                  aria-label={isInvitation ? "Copy invitation link" : "Copy password reset link"}
                  onClick={() => copyToClipboard(invitationUrl)}
                >
                  <Copy />
                  Copy
                </InputGroupButton>
              </InputGroupAddon>
            </InputGroup>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
