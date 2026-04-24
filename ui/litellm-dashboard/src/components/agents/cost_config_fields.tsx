import React from "react";
import { useFormContext } from "react-hook-form";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from "lucide-react";
import { AGENT_FORM_CONFIG } from "./agent_config";

const InfoTip: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <TooltipProvider>
    <Tooltip>
      <TooltipTrigger asChild>
        <Info className="ml-1 inline h-3 w-3 text-muted-foreground" />
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{children}</TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

const CostConfigFields: React.FC = () => {
  const { register } = useFormContext();
  return (
    <div className="space-y-4">
      {AGENT_FORM_CONFIG.cost.fields.map((field) => (
        <div key={field.name} className="space-y-2">
          <Label htmlFor={field.name}>
            {field.label}
            {field.tooltip ? <InfoTip>{field.tooltip}</InfoTip> : null}
          </Label>
          <Input
            id={field.name}
            placeholder={field.placeholder}
            type="number"
            step="0.000001"
            {...register(field.name)}
          />
        </div>
      ))}
    </div>
  );
};

export default CostConfigFields;
