import { useLanguage } from "@/contexts/LanguageContext";
import { Button } from "@/components/ui/button";
import { TriangleAlert } from "lucide-react";
import * as React from "react";

interface ErrorStateProps {
  title?: string;
  message?: React.ReactNode;
  onRetry?: () => void;
}

export function ErrorState({ title, message, onRetry }: ErrorStateProps) {
  const { t } = useLanguage();
  return (
    <div role="alert" className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-destructive/10">
        <TriangleAlert className="size-5 text-destructive" />
      </div>
      <div className="text-sm font-medium text-foreground">{title ?? t("Failed to load")}</div>
      {message ? <div className="max-w-prose text-center text-sm text-muted-foreground">{message}</div> : null}
      {onRetry ? (
        <Button variant="outline" size="sm" className="mt-2" onClick={onRetry}>
          {t("Retry")}
        </Button>
      ) : null}
    </div>
  );
}
