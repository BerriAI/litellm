import { useState } from "react";

import { Button } from "@/components/ui/button";

import { BulkInviteDialog } from "./BulkInviteDialog";

interface BulkInviteButtonProps {
  accessToken: string;
  teams: readonly { team_id: string }[] | null;
  possibleUIRoles: null | Record<string, Record<string, string>>;
  onUsersCreated?: () => void;
}

export function BulkInviteButton({ accessToken, teams, onUsersCreated }: BulkInviteButtonProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <Button className="mb-0" onClick={() => setIsOpen(true)}>
        + Bulk Invite Users
      </Button>
      <BulkInviteDialog
        open={isOpen}
        onClose={() => setIsOpen(false)}
        accessToken={accessToken}
        teams={teams}
        onUsersCreated={onUsersCreated}
      />
    </>
  );
}
