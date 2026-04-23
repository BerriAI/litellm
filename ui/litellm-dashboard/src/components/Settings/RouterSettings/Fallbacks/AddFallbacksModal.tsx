import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ArrowRight } from "lucide-react";
import React from "react";

interface AddFallbacksModalProps {
  open: boolean;
  onCancel: () => void;
  children: React.ReactNode;
}

export function AddFallbacksModal({
  open,
  onCancel,
  children,
}: AddFallbacksModalProps) {
  return (
    <Dialog open={open} onOpenChange={(o) => (!o ? onCancel() : undefined)}>
      <DialogContent className="max-w-[900px]">
        <DialogHeader>
          <div className="flex items-center gap-2 text-foreground">
            <div className="p-2 bg-indigo-50 dark:bg-indigo-950/30 rounded-lg">
              <ArrowRight className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
            </div>
            <div>
              <DialogTitle className="text-lg font-bold m-0">
                Configure Model Fallbacks
              </DialogTitle>
              <p className="text-sm text-muted-foreground font-normal m-0">
                Manage multiple fallback chains for different models (up to 5
                groups at a time)
              </p>
            </div>
          </div>
        </DialogHeader>
        <div className="mt-2">{children}</div>
      </DialogContent>
    </Dialog>
  );
}
