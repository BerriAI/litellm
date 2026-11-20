import type { FormEventHandler, ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

export function CustomerBudgetDialog({
  title,
  description,
  pending,
  onClose,
  children,
  open = true,
  className,
}: {
  title: string;
  description?: string;
  pending: boolean;
  onClose: () => void;
  children: ReactNode;
  open?: boolean;
  className?: string;
}) {
  return (
    <Dialog open={open} onOpenChange={(next) => !next && !pending && onClose()}>
      <DialogContent
        className={cn("flex max-h-[85vh] flex-col gap-0 overflow-hidden p-0 sm:max-w-4xl", className)}
        showCloseButton={!pending}
      >
        <DialogHeader className="shrink-0 border-b px-6 py-5">
          <DialogTitle className="text-xl">{title}</DialogTitle>
          {description && <DialogDescription>{description}</DialogDescription>}
        </DialogHeader>
        {children}
      </DialogContent>
    </Dialog>
  );
}

export function CustomerBudgetForm({
  pending,
  error,
  onSubmit,
  onClose,
  children,
  dirty = true,
  saveLabel,
  label,
}: {
  pending: boolean;
  error: string | null;
  onSubmit: FormEventHandler<HTMLFormElement>;
  onClose: () => void;
  children: ReactNode;
  dirty?: boolean;
  saveLabel?: string;
  label?: string;
}) {
  const { t } = useTranslation();
  return (
    <form className="flex min-h-0 flex-1 flex-col" onSubmit={onSubmit} aria-label={label}>
      <div className="min-h-0 overflow-y-auto px-6 py-6">
        <fieldset disabled={pending} className="min-w-0 space-y-5">
          {children}
        </fieldset>
      </div>
      <div className="shrink-0 border-t bg-muted/20 px-6 py-4">
        {error && (
          <p role="alert" className="mb-3 text-sm text-destructive">
            {error}
          </p>
        )}
        <DialogFooter>
          <Button type="button" variant="outline" disabled={pending} onClick={onClose}>
            {t("customerPortal.common.cancel")}
          </Button>
          <Button type="submit" disabled={pending || !dirty}>
            {saveLabel ?? t(pending ? "customerPortal.hierarchy.saving" : "customerPortal.hierarchy.save")}
          </Button>
        </DialogFooter>
      </div>
    </form>
  );
}
