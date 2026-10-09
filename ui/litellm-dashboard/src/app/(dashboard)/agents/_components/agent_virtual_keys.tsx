import React from "react";
import { KeyRound } from "lucide-react";
import { KeyResponse } from "@/components/key_team_helpers/key_list";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface AgentVirtualKeysProps {
  keys: KeyResponse[];
  isLoading: boolean;
  onKeyClick: (key: KeyResponse) => void;
}

const AgentVirtualKeys: React.FC<AgentVirtualKeysProps> = ({ keys, isLoading, onKeyClick }) => {
  return (
    <div className="mt-6">
      <h4 className="text-base font-semibold text-foreground">Virtual Keys</h4>
      {isLoading ? (
        <p className="mt-2 text-sm text-muted-foreground">Loading keys...</p>
      ) : keys.length === 0 ? (
        <p className="mt-2 text-sm text-muted-foreground">No virtual key assigned to this agent.</p>
      ) : (
        <div className="mt-3 flex flex-col gap-2">
          {keys.map((key) => (
            <div key={key.token} className="flex items-center gap-3 rounded-sm border border-border px-3 py-2">
              <KeyRound className="size-4 text-muted-foreground" />
              <span className="text-sm font-medium text-foreground">{key.key_alias || "Unnamed key"}</span>
              {key.key_name && <span className="font-mono text-xs text-muted-foreground">{key.key_name}</span>}
              <TooltipProvider delay={300}>
                <Tooltip>
                  <TooltipTrigger
                    render={
                      <Button variant="link" size="sm" className="ml-auto font-mono" onClick={() => onKeyClick(key)}>
                        {key.token?.slice(0, 12)}...
                      </Button>
                    }
                  />
                  <TooltipContent>{key.token}</TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default AgentVirtualKeys;
