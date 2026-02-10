import React, { useRef, useState } from "react";
import { Input, Tabs, Typography } from "antd";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import { CodeOutlined, EyeOutlined } from "@ant-design/icons";

const { TextArea } = Input;
const { Text } = Typography;

interface CustomCodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  height?: string;
  placeholder?: string;
  disabled?: boolean;
}

const CustomCodeEditor: React.FC<CustomCodeEditorProps> = ({
  value,
  onChange,
  height = "350px",
  placeholder = `def apply_guardrail(inputs, request_data, input_type):
    # inputs: contains texts, images, tools, tool_calls, structured_messages, model
    # request_data: contains model, user_id, team_id, end_user_id, metadata
    # input_type: "request" or "response"
    
    for text in inputs["texts"]:
        # Example: Block if SSN pattern is detected
        if regex_match(text, r"\\d{3}-\\d{2}-\\d{4}"):
            return block("SSN detected in message")
    
    return allow()`,
  disabled = false,
}) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [activeTab, setActiveTab] = useState<string>("edit");
  const [cursorPosition, setCursorPosition] = useState({ line: 1, column: 1 });

  // Calculate cursor position
  const updateCursorPosition = () => {
    if (textareaRef.current) {
      const textarea = textareaRef.current;
      const textBeforeCursor = value.substring(0, textarea.selectionStart);
      const lines = textBeforeCursor.split("\n");
      const line = lines.length;
      const column = lines[lines.length - 1].length + 1;
      setCursorPosition({ line, column });
    }
  };

  // Handle tab key for indentation
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Tab") {
      e.preventDefault();
      const textarea = e.currentTarget;
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      
      // Insert 4 spaces at cursor position
      const newValue = value.substring(0, start) + "    " + value.substring(end);
      onChange(newValue);
      
      // Move cursor after the inserted spaces
      setTimeout(() => {
        textarea.selectionStart = textarea.selectionEnd = start + 4;
      }, 0);
    }
  };

  const lineCount = value.split("\n").length;

  const tabItems = [
    {
      key: "edit",
      label: (
        <span className="flex items-center gap-1.5">
          <CodeOutlined />
          Edit
        </span>
      ),
      children: (
        <div className="relative" style={{ height }}>
          {/* Line numbers */}
          <div 
            className="absolute left-0 top-0 bottom-0 w-12 bg-[#1e1e1e] border-r border-[#3c3c3c] text-right pr-2 pt-3 overflow-hidden select-none"
            style={{ fontFamily: "monospace", fontSize: "13px", lineHeight: "1.5" }}
          >
            {Array.from({ length: Math.max(lineCount, 15) }, (_, i) => (
              <div key={i + 1} className="text-gray-500 h-[19.5px]">
                {i + 1}
              </div>
            ))}
          </div>
          
          {/* Code editor */}
          <textarea
            ref={textareaRef as any}
            value={value}
            onChange={(e) => {
              onChange(e.target.value);
              updateCursorPosition();
            }}
            onKeyDown={handleKeyDown}
            onClick={updateCursorPosition}
            onKeyUp={updateCursorPosition}
            placeholder={placeholder}
            disabled={disabled}
            spellCheck={false}
            className="w-full h-full pl-14 pr-4 pt-3 pb-3 font-mono text-sm resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
            style={{
              backgroundColor: "#1e1e1e",
              color: "#d4d4d4",
              border: "1px solid #3c3c3c",
              borderRadius: "8px",
              lineHeight: "1.5",
              tabSize: 4,
            }}
          />
          
          {/* Status bar */}
          <div className="absolute bottom-0 left-0 right-0 h-6 bg-[#252526] border-t border-[#3c3c3c] flex items-center justify-between px-3 text-xs text-gray-400 rounded-b-lg">
            <span>Python-like (Sandboxed)</span>
            <span>Ln {cursorPosition.line}, Col {cursorPosition.column}</span>
          </div>
        </div>
      ),
    },
    {
      key: "preview",
      label: (
        <span className="flex items-center gap-1.5">
          <EyeOutlined />
          Preview
        </span>
      ),
      children: (
        <div style={{ height }} className="overflow-auto rounded-lg border border-gray-200">
          <SyntaxHighlighter
            language="python"
            style={vscDarkPlus}
            showLineNumbers
            wrapLines
            customStyle={{
              margin: 0,
              borderRadius: "8px",
              fontSize: "13px",
              minHeight: height,
            }}
            lineNumberStyle={{
              minWidth: "3em",
              paddingRight: "1em",
              color: "#6e7681",
              borderRight: "1px solid #3c3c3c",
              marginRight: "1em",
            }}
          >
            {value || placeholder}
          </SyntaxHighlighter>
        </div>
      ),
    },
  ];

  return (
    <div className="custom-code-editor">
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={tabItems}
        className="custom-code-tabs"
        size="small"
      />
      <style>{`
        .custom-code-tabs .ant-tabs-nav {
          margin-bottom: 8px;
        }
        .custom-code-tabs .ant-tabs-tab {
          padding: 4px 12px;
        }
        .custom-code-editor textarea::placeholder {
          color: #6e7681;
        }
      `}</style>
    </div>
  );
};

export default CustomCodeEditor;
