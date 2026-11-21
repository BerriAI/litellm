import React from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

const BlockedReasonTooltip: React.FC<{ reason: string | null; children: React.ReactElement }> = ({
  reason,
  children,
}) =>
  reason === null ? (
    children
  ) : (
    <Tooltip>
      <TooltipTrigger render={children} />
      <TooltipContent>{reason}</TooltipContent>
    </Tooltip>
  );

export default BlockedReasonTooltip;
