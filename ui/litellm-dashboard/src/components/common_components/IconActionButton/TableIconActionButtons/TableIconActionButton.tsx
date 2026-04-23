import {
  ChevronDown,
  ChevronUp,
  Copy,
  ExternalLink,
  Pencil,
  Play,
  RefreshCw,
  Trash2,
} from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import BaseActionButton from "../BaseActionButton";

export interface TableIconActionButtonProps {
  onClick: () => void;
  tooltipText?: string;
  disabled?: boolean;
  disabledTooltipText?: string;
  dataTestId?: string;
  variant: keyof typeof TableIconActionButtonMap;
}

export interface TableIconActionButtonBaseProps {
  // Accepts both lucide-react ForwardRefExoticComponent and plain SVG
  // function components.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  icon: React.ComponentType<any>;
  className?: string;
}

// Categorical hover-color palette: each action variant has its own
// hover-tint. Documented in DEVIATIONS.md alongside the policy/status
// palette decisions.
export const TableIconActionButtonMap: Record<
  string,
  TableIconActionButtonBaseProps
> = {
  Edit: { icon: Pencil, className: "hover:text-blue-600" },
  Delete: { icon: Trash2, className: "hover:text-red-600" },
  Test: { icon: Play, className: "hover:text-blue-600" },
  Regenerate: { icon: RefreshCw, className: "hover:text-green-600" },
  Up: { icon: ChevronUp, className: "hover:text-blue-600" },
  Down: { icon: ChevronDown, className: "hover:text-blue-600" },
  Open: { icon: ExternalLink, className: "hover:text-green-600" },
  Copy: { icon: Copy, className: "hover:text-blue-600" },
};

export default function TableIconActionButton({
  onClick,
  tooltipText,
  disabled = false,
  disabledTooltipText,
  dataTestId,
  variant,
}: TableIconActionButtonProps) {
  const { icon, className } = TableIconActionButtonMap[variant];
  const message = disabled ? disabledTooltipText : tooltipText;
  const button = (
    <BaseActionButton
      icon={icon}
      onClick={onClick}
      className={className}
      disabled={disabled}
      dataTestId={dataTestId}
    />
  );
  if (!message) return button;
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <span>{button}</span>
        </TooltipTrigger>
        <TooltipContent>{message}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
