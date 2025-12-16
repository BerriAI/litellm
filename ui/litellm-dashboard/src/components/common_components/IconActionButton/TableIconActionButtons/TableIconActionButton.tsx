import {
  PencilAltIcon,
  PlayIcon,
  RefreshIcon,
  TrashIcon,
  ChevronUpIcon,
  ChevronDownIcon,
  ExternalLinkIcon,
} from "@heroicons/react/outline";
import { Tooltip } from "antd";
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
  Edit: { icon: PencilAltIcon, className: "hover:text-blue-600" },
  Delete: { icon: TrashIcon, className: "hover:text-red-600" },
  Test: { icon: PlayIcon, className: "hover:text-blue-600" },
  Regenerate: { icon: RefreshIcon, className: "hover:text-green-600" },
  Up: { icon: ChevronUpIcon, className: "hover:text-blue-600" },
  Down: { icon: ChevronDownIcon, className: "hover:text-blue-600" },
  Open: { icon: ExternalLinkIcon, className: "hover:text-green-600" },
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
  return (
    <Tooltip title={disabled ? disabledTooltipText : tooltipText}>
      <span>
        <BaseActionButton
          icon={icon}
          onClick={onClick}
          className={className}
          disabled={disabled}
          dataTestId={dataTestId}
        />
      </span>
    </Tooltip>
  );
}
