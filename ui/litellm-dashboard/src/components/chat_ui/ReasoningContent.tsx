import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";
import { ChevronDown, ChevronRight, Lightbulb } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";

interface ReasoningContentProps {
  reasoningContent: string;
}

const ReasoningContent: React.FC<ReasoningContentProps> = ({ reasoningContent }) => {
  const [isExpanded, setIsExpanded] = useState(true);

  if (!reasoningContent) return null;

  return (
    <div className="reasoning-content mt-1 mb-2">
      <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
        <CollapsibleTrigger
          render={
            <Button type="button" variant="ghost" size="sm" className="text-xs text-gray-500 hover:text-gray-700" />
          }
        >
          <Lightbulb className="size-3.5" />
          {isExpanded ? "Hide reasoning" : "Show reasoning"}
          {isExpanded ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
        </CollapsibleTrigger>

        <CollapsibleContent>
          <div
            className="mt-2 max-w-full overflow-x-auto whitespace-pre-wrap break-words rounded-md border border-gray-200 bg-gray-50 p-3 text-sm text-gray-700"
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
                      style={coy as { [key: string]: React.CSSProperties }}
                    >
                      {String(children).replace(/\n$/, "")}
                    </SyntaxHighlighter>
                  ) : (
                    <code
                      className={`${className ?? ""} rounded-sm bg-gray-100 px-1.5 py-0.5 font-mono text-sm`}
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
