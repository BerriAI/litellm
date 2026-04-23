import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";
import { ChevronDown, ChevronRight, Lightbulb } from "lucide-react";

interface ReasoningContentProps {
  reasoningContent: string;
}

const ReasoningContent: React.FC<ReasoningContentProps> = ({
  reasoningContent,
}) => {
  const [isExpanded, setIsExpanded] = useState(true);

  if (!reasoningContent) return null;

  return (
    <div className="reasoning-content mt-1 mb-2">
      <Button
        variant="ghost"
        size="sm"
        className="flex items-center text-xs text-muted-foreground hover:text-foreground"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <Lightbulb className="h-3.5 w-3.5" />
        {isExpanded ? "Hide reasoning" : "Show reasoning"}
        {isExpanded ? (
          <ChevronDown className="h-3 w-3 ml-1" />
        ) : (
          <ChevronRight className="h-3 w-3 ml-1" />
        )}
      </Button>

      {isExpanded && (
        <div className="mt-2 p-3 bg-muted border border-border rounded-md text-sm text-foreground">
          <ReactMarkdown
            components={{
              code({
                inline,
                className,
                children,
                ...props
              }: React.ComponentPropsWithoutRef<"code"> & {
                inline?: boolean;
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                node?: any;
              }) {
                const match = /language-(\w+)/.exec(className || "");
                return !inline && match ? (
                  <SyntaxHighlighter
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    style={coy as any}
                    language={match[1]}
                    PreTag="div"
                    className="rounded-md my-2"
                    {...props}
                  >
                    {String(children).replace(/\n$/, "")}
                  </SyntaxHighlighter>
                ) : (
                  <code
                    className={`${className} px-1.5 py-0.5 rounded bg-muted text-sm font-mono`}
                    {...props}
                  >
                    {children}
                  </code>
                );
              },
            }}
          >
            {reasoningContent}
          </ReactMarkdown>
        </div>
      )}
    </div>
  );
};

export default ReasoningContent;
