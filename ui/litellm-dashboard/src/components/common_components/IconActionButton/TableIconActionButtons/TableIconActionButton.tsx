import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import {
  PencilAltIcon,
  PlayIcon,
  RefreshIcon,
  TrashIcon,
  ChevronUpIcon,
  ChevronDownIcon,
  ExternalLinkIcon,
  ClipboardCopyIcon,
} from "@heroicons/react/outline";
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
  icon: React.ComponentType<React.ComponentProps<"svg">>;
  className?: string;
}

export const TableIconActionButtonMap: Record<string, TableIconActionButtonBaseProps> = {
  Edit: { icon: PencilAltIcon, className: "hover:text-info" },
  Delete: { icon: TrashIcon, className: "hover:text-destructive" },
  Test: { icon: PlayIcon, className: "hover:text-info" },
  Regenerate: { icon: RefreshIcon, className: "hover:text-success" },
  Up: { icon: ChevronUpIcon, className: "hover:text-info" },
  Down: { icon: ChevronDownIcon, className: "hover:text-info" },
  Open: { icon: ExternalLinkIcon, className: "hover:text-success" },
  Copy: { icon: ClipboardCopyIcon, className: "hover:text-info" },
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
  const title = disabled ? disabledTooltipText : tooltipText;
  const button = (
    <BaseActionButton icon={icon} onClick={onClick} className={className} disabled={disabled} dataTestId={dataTestId} />
  );

  if (!title) {
    return <span>{button}</span>;
  }

  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger render={<span />}>{button}</TooltipTrigger>
        <TooltipContent>{title}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
