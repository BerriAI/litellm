import { useDisableShowPrompts } from "@/app/(dashboard)/hooks/useDisableShowPrompts";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Github, Slack } from "lucide-react";
import React from "react";

const iconBtnClass =
  "inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md border-0 bg-transparent text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-700 cursor-pointer";

export const CommunityEngagementButtons: React.FC = () => {
  const disableShowPrompts = useDisableShowPrompts();

  if (disableShowPrompts) {
    return null;
  }

  return (
    <div
      className="flex items-center gap-0.5 rounded-md border border-gray-200/80 bg-gray-50 px-0.5 py-0"
      aria-label="Community links"
    >
      <TooltipProvider>
        <Tooltip>
          <TooltipTrigger
            render={
              <a
                href="https://www.litellm.ai/support"
                target="_blank"
                rel="noopener noreferrer"
                className={iconBtnClass}
                aria-label="Join Slack"
              />
            }
          >
            <Slack className="size-[18px]" />
          </TooltipTrigger>
          <TooltipContent>LiteLLM Slack community</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger
            render={
              <a
                href="https://github.com/BerriAI/litellm"
                target="_blank"
                rel="noopener noreferrer"
                className={iconBtnClass}
                aria-label="LiteLLM on GitHub"
              />
            }
          >
            <Github className="size-[18px]" />
          </TooltipTrigger>
          <TooltipContent>LiteLLM on GitHub</TooltipContent>
        </Tooltip>
      </TooltipProvider>
    </div>
  );
};
