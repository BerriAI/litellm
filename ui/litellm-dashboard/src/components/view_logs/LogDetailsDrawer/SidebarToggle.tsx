import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";

export interface SidebarToggleProps {
  isCollapsed: boolean;
  onToggle: () => void;
}

export function SidebarToggle({ isCollapsed, onToggle }: SidebarToggleProps) {
  return (
    <Button
      variant="ghost"
      size="icon-xs"
      onClick={onToggle}
      className="shrink-0"
      aria-label={isCollapsed ? "Expand trace sidebar" : "Collapse trace sidebar"}
    >
      {isCollapsed ? <ChevronRight className="size-4" /> : <ChevronLeft className="size-4" />}
    </Button>
  );
}
