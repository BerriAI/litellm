import React, { useState } from "react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Badge } from "@/components/ui/badge";
import { Eye, Loader2 } from "lucide-react";
import { PolicyAttachment } from "./types";
import { estimateAttachmentImpactCall } from "../networking";

interface ImpactInfo {
  affected_keys_count: number;
  affected_teams_count: number;
  sample_keys: string[];
  sample_teams: string[];
}

const ImpactPopover: React.FC<{
  attachment: PolicyAttachment;
  accessToken: string | null;
}> = ({ attachment, accessToken }) => {
  const [impact, setImpact] = useState<ImpactInfo | null>(null);
  const [loading, setLoading] = useState(false);
  const [loaded, setLoaded] = useState(false);

  const loadImpact = async () => {
    if (loaded || loading || !accessToken) return;
    setLoading(true);
    try {
      const data = await estimateAttachmentImpactCall(accessToken, {
        policy_name: attachment.policy_name,
        scope: attachment.scope,
        teams: attachment.teams,
        keys: attachment.keys,
        models: attachment.models,
        tags: attachment.tags,
      });
      setImpact(data);
      setLoaded(true);
    } catch (error) {
      console.error("Failed to load impact:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Popover
      onOpenChange={(open) => {
        if (open) loadImpact();
      }}
    >
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger asChild>
            <PopoverTrigger asChild>
              <button
                type="button"
                className="inline-flex items-center justify-center cursor-pointer text-muted-foreground hover:text-primary"
                aria-label="View blast radius"
              >
                <Eye className="h-3.5 w-3.5" />
              </button>
            </PopoverTrigger>
          </TooltipTrigger>
          <TooltipContent>View blast radius</TooltipContent>
        </Tooltip>
      </TooltipProvider>
      <PopoverContent className="max-w-[320px]">
        <div className="font-semibold text-sm mb-2">Blast Radius</div>
        {loading ? (
          <div className="p-2 text-center flex items-center justify-center gap-2 text-sm">
            <Loader2 className="h-3 w-3 animate-spin" /> Loading...
          </div>
        ) : impact ? (
          <div className="text-xs">
            {impact.affected_keys_count === -1 ? (
              <p className="font-medium text-amber-600 dark:text-amber-400">
                Global scope — affects all keys and teams
              </p>
            ) : (
              <>
                <p className="mb-1">
                  <strong>{impact.affected_keys_count}</strong> key
                  {impact.affected_keys_count !== 1 ? "s" : ""},{" "}
                  <strong>{impact.affected_teams_count}</strong> team
                  {impact.affected_teams_count !== 1 ? "s" : ""} affected
                </p>
                {impact.sample_keys.length > 0 && (
                  <div className="mb-1">
                    <span className="text-muted-foreground">Keys: </span>
                    {impact.sample_keys.map((k: string) => (
                      <Badge
                        key={k}
                        variant="outline"
                        className="text-[10px] m-0.5"
                      >
                        {k}
                      </Badge>
                    ))}
                  </div>
                )}
                {impact.sample_teams.length > 0 && (
                  <div>
                    <span className="text-muted-foreground">Teams: </span>
                    {impact.sample_teams.map((t: string) => (
                      <Badge
                        key={t}
                        variant="outline"
                        className="text-[10px] m-0.5"
                      >
                        {t}
                      </Badge>
                    ))}
                  </div>
                )}
                {impact.affected_keys_count === 0 &&
                  impact.affected_teams_count === 0 && (
                    <p className="text-muted-foreground">
                      No keys or teams currently affected
                    </p>
                  )}
              </>
            )}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">Click to load</p>
        )}
      </PopoverContent>
    </Popover>
  );
};

export default ImpactPopover;
