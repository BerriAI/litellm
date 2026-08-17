import React from "react";

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
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

const EditLoggingCredentialModal: React.FC<EditLoggingCredentialModalProps> = ({
  accessToken,
  credentialName,
  access,
  credentialInfo,
  open,
  onClose,
  onSaved,
}) => {
  // The caller keys this component on the destination name, so each open remounts
  // and seeds from that row's access rather than syncing a prop into state.
  const [draft, setDraft] = React.useState<CredentialAccess>(access ?? {});

  const handleSave = async () => {
    if (!credentialName) return;
    // Always send the full access object; a global grant supersedes team/org.
    const next: CredentialAccess = draft.global
      ? { global: true, teams: [], orgs: [] }
      : { global: false, teams: draft.teams ?? [], orgs: draft.orgs ?? [] };
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
    <Dialog open={open} onOpenChange={(next) => !next && onClose()}>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>{credentialName ? `Edit scope: ${credentialName}` : "Edit scope"}</DialogTitle>
        </DialogHeader>
        <div className="space-y-4">
          <AccessControlFields value={draft} onChange={setDraft} />
        </div>
        <DialogFooter>
          <Button type="button" variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button type="button" onClick={handleSave}>
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default EditLoggingCredentialModal;
