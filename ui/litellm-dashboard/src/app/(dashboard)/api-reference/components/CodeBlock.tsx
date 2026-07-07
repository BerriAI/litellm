"use client";

import { useState } from "react";
import { CheckIcon, ClipboardIcon } from "lucide-react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import { Button } from "@/components/ui/button";

interface CodeBlockProps {
  code: string;
  language: string;
}

const CodeBlock = ({ code, language }: CodeBlockProps) => {
  const [copied, setCopied] = useState(false);
  const copyToClipboard = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="relative rounded-lg border border-border overflow-hidden bg-muted/30">
      <Button
        variant="ghost"
        size="icon-xs"
        onClick={copyToClipboard}
        className="absolute top-3 right-3 z-10"
        aria-label="Copy code"
      >
        {copied ? <CheckIcon className="size-3.5" /> : <ClipboardIcon className="size-3.5" />}
      </Button>
      <SyntaxHighlighter
        language={language}
        style={oneLight}
        customStyle={{
          margin: 0,
          padding: "1.5rem",
          borderRadius: "0.5rem",
          fontSize: "0.8rem",
          backgroundColor: "transparent",
        }}
        showLineNumbers
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
};

export default CodeBlock;
