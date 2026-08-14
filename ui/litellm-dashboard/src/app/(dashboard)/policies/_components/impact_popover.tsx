import React, { useState } from "react";
import { Eye, Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Popover, PopoverContent, PopoverTitle, PopoverTrigger } from "@/components/ui/popover";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { PolicyAttachment } from "@/components/policies/types";
import { estimateAttachmentImpactCall } from "@/components/networking";

interface ImpactResult {
  affected_keys_count: number;
  affected_teams_count: number;
  sample_keys: string[];
  sample_teams: string[];
}

const ImpactPopover: React.FC<{ attachment: PolicyAttachment; accessToken: string | null }> = ({
  attachment,
  accessToken,
}) => {
  const [impact, setImpact] = useState<ImpactResult | null>(null);
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
          <TooltipTrigger
            render={
              <PopoverTrigger
                render={
                  <Button variant="ghost" size="icon-xs" aria-label="View blast radius">
                    <Eye />
                  </Button>
                }
              />
            }
          />
          <TooltipContent>View blast radius</TooltipContent>
        </Tooltip>
      </TooltipProvider>

      <PopoverContent className="w-72 gap-2">
        <PopoverTitle>Blast Radius</PopoverTitle>
        {loading ? (
          <div className="flex items-center justify-center gap-2 py-2 text-xs text-muted-foreground">
            <Loader2 className="size-3.5 animate-spin" aria-hidden="true" />
            Loading...
          </div>
        ) : impact ? (
          <div className="text-xs">
            {impact.affected_keys_count === -1 ? (
              <p className="font-medium text-foreground">Global scope — affects all keys and teams</p>
            ) : (
              <>
                <p className="mb-1">
                  <strong>{impact.affected_keys_count}</strong> key{impact.affected_keys_count !== 1 ? "s" : ""},{" "}
                  <strong>{impact.affected_teams_count}</strong> team{impact.affected_teams_count !== 1 ? "s" : ""}{" "}
                  affected
                </p>
                {impact.sample_keys.length > 0 && (
                  <div className="mb-1 flex flex-wrap items-center gap-1">
                    <span className="text-muted-foreground">Keys:</span>
                    {impact.sample_keys.map((key: string) => (
                      <Badge key={key} variant="secondary" className="px-1.5 py-0 text-[10px] font-normal">
                        {key}
                      </Badge>
                    ))}
                  </div>
                )}
                {impact.sample_teams.length > 0 && (
                  <div className="flex flex-wrap items-center gap-1">
                    <span className="text-muted-foreground">Teams:</span>
                    {impact.sample_teams.map((team: string) => (
                      <Badge key={team} variant="secondary" className="px-1.5 py-0 text-[10px] font-normal">
                        {team}
                      </Badge>
                    ))}
                  </div>
                )}
                {impact.affected_keys_count === 0 && impact.affected_teams_count === 0 && (
                  <p className="text-muted-foreground">No keys or teams currently affected</p>
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
