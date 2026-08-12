"use client";

import React, { useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";

export type ReviewDecision = "approve" | "reject";

interface ReviewSkillDialogProps {
  skillName: string;
  decision: ReviewDecision;
  isSubmitting: boolean;
  onCancel: () => void;
  onConfirm: (notes: string) => void;
}

const ReviewSkillDialog: React.FC<ReviewSkillDialogProps> = ({
  skillName,
  decision,
  isSubmitting,
  onCancel,
  onConfirm,
}) => {
  const [notes, setNotes] = useState("");
  const approving = decision === "approve";

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onCancel();
      }}
    >
      <DialogContent data-testid="review-skill-dialog">
        <DialogHeader>
          <DialogTitle>{approving ? "Approve skill" : "Reject skill"}</DialogTitle>
          <DialogDescription>
            {approving
              ? `Approving "${skillName}" publishes it to the Skill Hub and marketplace.json for all users.`
              : `Rejecting "${skillName}" keeps it unpublished. The submitter sees your notes.`}
          </DialogDescription>
        </DialogHeader>
        {!approving && (
          <div className="flex flex-col gap-2">
            <Label htmlFor="review-notes">Reason for rejection (optional)</Label>
            <Textarea
              id="review-notes"
              data-testid="review-notes"
              value={notes}
              onChange={(event) => setNotes(event.target.value)}
              placeholder="Point to the repository path that needs fixing"
            />
          </div>
        )}
        <DialogFooter>
          <Button variant="secondary" onClick={onCancel} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button
            variant={approving ? "default" : "destructive"}
            data-testid="review-confirm"
            onClick={() => onConfirm(notes)}
            disabled={isSubmitting}
          >
            {approving ? "Approve" : "Reject"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default ReviewSkillDialog;
