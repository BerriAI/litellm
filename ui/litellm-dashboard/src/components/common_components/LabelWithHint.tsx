import React from "react";

import { SimpleTooltip } from "@/components/ui/tooltip";

/**
 * antd's `tooltip=` prop renders a hover hint beside the label, which is a different channel from
 * `help=`/`extra=` and must not collapse into always-visible description text.
 */
export const labelWithHint = (label: React.ReactNode, hint?: React.ReactNode): React.ReactNode =>
  hint === undefined || hint === null || hint === "" ? (
    label
  ) : (
    <>
      {label}
      <SimpleTooltip content={hint} />
    </>
  );
