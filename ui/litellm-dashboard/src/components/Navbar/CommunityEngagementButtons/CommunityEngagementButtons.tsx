import { useDisableShowPrompts } from "@/app/(dashboard)/hooks/useDisableShowPrompts";
import { Button } from "@/components/ui/button";
import { Github, Slack } from "lucide-react";
import React from "react";

export const CommunityEngagementButtons: React.FC = () => {
  const disableShowPrompts = useDisableShowPrompts();

  if (disableShowPrompts) {
    return null;
  }

  return (
    <>
      <Button
        asChild
        variant="outline"
        className="shadow-md shadow-indigo-500/20 hover:shadow-indigo-500/50 transition-shadow"
      >
        <a
          href="https://www.litellm.ai/support"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Slack className="h-4 w-4" />
          Join Slack
        </a>
      </Button>
      <Button
        asChild
        variant="outline"
        className="shadow-md shadow-indigo-500/20 hover:shadow-indigo-500/50 transition-shadow"
      >
        <a
          href="https://github.com/BerriAI/litellm"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Github className="h-4 w-4" />
          Star us on GitHub
        </a>
      </Button>
    </>
  );
};
