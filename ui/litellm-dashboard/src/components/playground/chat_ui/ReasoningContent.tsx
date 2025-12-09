import React, { useState } from "react";
import { Button } from "antd";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { coy } from "react-syntax-highlighter/dist/esm/styles/prism";
import { DownOutlined, RightOutlined, BulbOutlined } from "@ant-design/icons";

interface ReasoningContentProps {
  reasoningContent: string;
}

const ReasoningContent: React.FC<ReasoningContentProps> = ({ reasoningContent }) => {
  const [isExpanded, setIsExpanded] = useState(true);

  if (!reasoningContent) return null;

  return (
    <div className="reasoning-content mt-1 mb-2">
      <Button
        type="text"
        className="flex items-center text-xs text-gray-500 hover:text-gray-700"
        onClick={() => setIsExpanded(!isExpanded)}
        icon={<BulbOutlined />}
      >
        {isExpanded ? "Hide reasoning" : "Show reasoning"}
        {isExpanded ? <DownOutlined className="ml-1" /> : <RightOutlined className="ml-1" />}
      </Button>

      {isExpanded && (
        <div className="mt-2 p-3 bg-gray-50 border border-gray-200 rounded-md text-sm text-gray-700">
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
                node?: any;
              }) {
                const match = /language-(\w+)/.exec(className || "");
                return !inline && match ? (
                  <SyntaxHighlighter
                    style={coy as any}
                    language={match[1]}
                    PreTag="div"
                    className="rounded-md my-2"
                    {...props}
                  >
                    {String(children).replace(/\n$/, "")}
                  </SyntaxHighlighter>
                ) : (
                  <code className={`${className} px-1.5 py-0.5 rounded bg-gray-100 text-sm font-mono`} {...props}>
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
