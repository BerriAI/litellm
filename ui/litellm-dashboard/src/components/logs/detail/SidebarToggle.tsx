import { ChevronLeft, ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/cva.config";

export interface SidebarToggleProps {
  isCollapsed: boolean;
  onToggle: () => void;
  className?: string;
}

export function SidebarToggle({ isCollapsed, onToggle, className }: SidebarToggleProps) {
  const { t } = useTranslation();
  return (
    <Button
      variant="ghost"
      size="icon-sm"
      onClick={onToggle}
      className={cn("shrink-0 bg-card! border! border-border! rounded-md!", className)}
      aria-label={isCollapsed ? t("logs.detail.sidebarToggle.expand") : t("logs.detail.sidebarToggle.collapse")}
    >
      {isCollapsed ? <ChevronLeft className="size-4" /> : <ChevronRight className="size-4" />}
    </Button>
  );
}
