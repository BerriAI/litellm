import { useState } from "react";
import { CheckIcon, ClipboardIcon } from "lucide-react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";

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
    <div className="relative rounded-lg border border-gray-200 overflow-hidden">
      <button
        onClick={copyToClipboard}
        className="absolute top-3 right-3 p-2 rounded-md bg-gray-100 hover:bg-gray-200 text-gray-600 z-10"
        aria-label="Copy code"
      >
        {copied ? <CheckIcon size={16} /> : <ClipboardIcon size={16} />}
      </button>
      <SyntaxHighlighter
        language={language}
        style={oneLight}
        customStyle={{
          margin: 0,
          padding: "1.5rem",
          borderRadius: "0.5rem",
          fontSize: "0.9rem",
          backgroundColor: "#fafafa",
        }}
        showLineNumbers
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
};

export default CodeBlock;
