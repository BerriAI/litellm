/**
 * Modal wrapper for the fallback selection form
 * Handles modal visibility and layout, but delegates content to children
 */

import { ArrowRight } from "lucide-react";
import React from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface AddFallbacksModalProps {
  open: boolean;
  onCancel: () => void;
  children: React.ReactNode;
}

export function AddFallbacksModal({ open, onCancel, children }: AddFallbacksModalProps) {
  return (
    <Dialog open={open} onOpenChange={(open) => !open && onCancel()} disablePointerDismissal>
      <DialogContent className="top-8 max-h-[calc(100dvh-4rem)] translate-y-0 overflow-y-auto sm:max-w-[900px]">
        <DialogHeader>
          <div className="pb-4 border-b border-border">
            <div className="flex items-center gap-2 text-foreground">
              <div className="p-2 bg-indigo-50 rounded-lg dark:bg-indigo-950">
                <ArrowRight className="w-5 h-5 text-indigo-600 dark:text-indigo-300" />
              </div>
              <div>
                <DialogTitle className="text-lg font-bold m-0">Configure Model Fallbacks</DialogTitle>
                <p className="text-sm text-muted-foreground font-normal m-0">
                  Manage multiple fallback chains for different models (up to 5 groups at a time)
                </p>
              </div>
            </div>
          </div>
        </DialogHeader>
        <div className="mt-6">{children}</div>
      </DialogContent>
    </Dialog>
  );
}
