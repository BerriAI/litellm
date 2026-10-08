import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";

import { useSyntaxTheme } from "@/hooks/useSyntaxTheme";
import { ChevronDown, ChevronRight, Lightbulb } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

interface ReasoningContentProps {
  reasoningContent: string;
}

const ReasoningContent: React.FC<ReasoningContentProps> = ({ reasoningContent }) => {
  const syntaxTheme = useSyntaxTheme(coy);
  const [isExpanded, setIsExpanded] = useState(true);

  if (!reasoningContent) return null;

  return (
    <div className="reasoning-content mt-1 mb-2">
      <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
        <CollapsibleTrigger
          render={
            <Button
              type="button"
              variant="ghost"
              size="sm"
              className="text-xs text-muted-foreground hover:text-foreground"
            />
          }
        >
          <Lightbulb className="size-3.5" />
          {isExpanded ? "Hide reasoning" : "Show reasoning"}
          {isExpanded ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
        </CollapsibleTrigger>

        <CollapsibleContent>
          <div
            className="mt-2 max-w-full overflow-x-auto whitespace-pre-wrap break-words rounded-md border border-border bg-muted p-3 text-sm text-foreground"
            style={{ wordBreak: "break-word", overflowWrap: "break-word" }}
          >
            <ReactMarkdown
              components={{
                code({
                  node,
                  inline,
                  className,
                  children,
                  ...props
                }: React.ComponentPropsWithoutRef<"code"> & {
                  inline?: boolean;
                  node?: unknown;
                }) {
                  const match = /language-(\w+)/.exec(className || "");
                  return !inline && match ? (
                    <SyntaxHighlighter
                      language={match[1]}
                      PreTag="div"
                      className="my-2 rounded-md"
                      wrapLines={true}
                      wrapLongLines={true}
                      {...props}
                      style={syntaxTheme}
                    >
                      {String(children).replace(/\n$/, "")}
                    </SyntaxHighlighter>
                  ) : (
                    <code
                      className={`${className ?? ""} rounded-sm bg-muted px-1.5 py-0.5 font-mono text-sm`}
                      style={{ wordBreak: "break-word" }}
                      {...props}
                    >
                      {children}
                    </code>
                  );
                },
                pre: ({ node, ...props }) => <pre style={{ overflowX: "auto", maxWidth: "100%" }} {...props} />,
              }}
            >
              {reasoningContent}
            </ReactMarkdown>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
};

export default ReasoningContent;
