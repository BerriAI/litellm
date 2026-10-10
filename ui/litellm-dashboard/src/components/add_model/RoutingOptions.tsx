import React from "react";
import { ChevronRight } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

interface RoutingOptionsProps {
  showValidationErrors?: boolean;
  summary?: string;
  children: React.ReactNode;
}

const RoutingOptions = ({ showValidationErrors = false, summary, children }: RoutingOptionsProps) => {
  const [open, setOpen] = React.useState(false);
  const [previousValidation, setPreviousValidation] = React.useState(showValidationErrors);
  if (previousValidation !== showValidationErrors) {
    setPreviousValidation(showValidationErrors);
    if (showValidationErrors) setOpen(true);
  }
  return (
    <Collapsible open={open} onOpenChange={setOpen} className="rounded-lg border">
      <CollapsibleTrigger className="group flex w-full flex-wrap items-center gap-2 px-4 py-3 text-left font-medium">
        <ChevronRight className="size-4 transition-transform group-data-panel-open:rotate-90" />
        Advanced settings
        {summary && <span className="text-xs font-normal text-muted-foreground">{summary}</span>}
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-4">{children}</CollapsibleContent>
    </Collapsible>
  );
};

export default RoutingOptions;
