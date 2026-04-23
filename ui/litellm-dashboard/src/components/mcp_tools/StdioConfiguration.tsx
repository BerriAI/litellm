import React from "react";
import { Form } from "antd";
import { Textarea } from "@/components/ui/textarea";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Info } from "lucide-react";

interface StdioConfigurationProps {
  isVisible: boolean;
  /**
   * When true, stdio_config is required + validated as JSON.
   * Edit screen can set this to false when using dedicated command/args/env fields.
   */
  required?: boolean;
}

const StdioConfiguration: React.FC<StdioConfigurationProps> = ({
  isVisible,
  required = true,
}) => {
  if (!isVisible) return null;

  return (
    <Form.Item
      label={
        <span className="text-sm font-medium text-foreground flex items-center">
          Stdio Configuration (JSON)
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="ml-2 h-3 w-3 inline text-primary cursor-help" />
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
                Paste your stdio MCP server configuration in JSON format. You
                can use the full mcpServers structure from config.yaml or just
                the inner server configuration.
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </span>
      }
      name="stdio_config"
      rules={[
        ...(required
          ? [{ required: true, message: "Please enter stdio configuration" }]
          : []),
        {
          validator: (_, value) => {
            if (!value) return Promise.resolve();
            try {
              JSON.parse(value);
              return Promise.resolve();
            } catch {
              return Promise.reject("Please enter valid JSON");
            }
          },
        },
      ]}
    >
      <Textarea
        placeholder={`{
  "mcpServers": {
    "circleci-mcp-server": {
      "command": "npx",
      "args": ["-y", "@circleci/mcp-server-circleci"],
      "env": {
        "CIRCLECI_TOKEN": "your-circleci-token",
        "CIRCLECI_BASE_URL": "https://circleci.com"
      }
    }
  }
}`}
        rows={12}
        className="rounded-lg font-mono text-sm"
      />
    </Form.Item>
  );
};

export default StdioConfiguration;
