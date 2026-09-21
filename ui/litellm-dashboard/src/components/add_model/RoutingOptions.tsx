import React from "react";
import { ChevronRight } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

interface RoutingOptionsProps {
  forecast: boolean;
  children: React.ReactNode;
}

const RoutingOptions = ({ forecast, children }: RoutingOptionsProps) =>
  forecast ? (
    <Collapsible className="rounded-lg border">
      <CollapsibleTrigger className="group flex w-full items-center gap-2 px-4 py-3 text-left font-medium">
        <ChevronRight className="size-4 transition-transform group-data-panel-open:rotate-90" />
        Advanced routing options
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-4 px-4 pb-4">{children}</CollapsibleContent>
    </Collapsible>
  ) : (
    <>{children}</>
  );

export default RoutingOptions;
