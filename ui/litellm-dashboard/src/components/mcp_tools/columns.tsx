import React from "react";
import { ColumnDef } from "@tanstack/react-table";
import { MCPTool, InputSchema } from "./types";
import { Button } from "@tremor/react"

export const columns: ColumnDef<MCPTool>[] = [
  {
    accessorKey: "mcp_info.server_name",
    header: "Provider",
    cell: ({ row }) => {
      const serverName = row.original.mcp_info.server_name;
      const logoUrl = row.original.mcp_info.logo_url;
      
      return (
        <div className="flex items-center space-x-2">
          {logoUrl && (
            <img 
              src={logoUrl} 
              alt={`${serverName} logo`} 
              className="h-5 w-5 object-contain"
            />
          )}
          <span className="font-medium">{serverName}</span>
        </div>
      );
    },
  },
  {
    accessorKey: "name",
    header: "Tool Name",
    cell: ({ row }) => {
      const name = row.getValue("name") as string;
      return (
        <div>
          <span className="font-mono text-sm">{name}</span>
        </div>
      );
    },
  },
  {
    accessorKey: "description",
    header: "Description",
    cell: ({ row }) => {
      const description = row.getValue("description") as string;
      return (
        <div className="max-w-md">
          <span className="text-sm text-gray-700">{description}</span>
        </div>
      );
    },
  },
  {
    id: "actions",
    header: "Actions",
    cell: ({ row }) => {
      const tool = row.original;
      
      return (
        <div className="flex items-center space-x-2">
          <Button 
            size="xs"
            variant="light"
            className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
            onClick={() => {
            if (typeof row.original.onToolSelect === 'function') {
                row.original.onToolSelect(tool);
            }
            }}
          >
            Test Tool
          </Button>
        </div>
      );
    },
  },
];

// Tool Panel component to display when a tool is selected
export function ToolTestPanel({
  tool,
  onSubmit,
  isLoading,
  result,
  error,
  onClose
}: {
  tool: MCPTool;
  onSubmit: (args: Record<string, any>) => void;
  isLoading: boolean;
  result: any | null;
  error: Error | null;
  onClose: () => void;
}) {
  const [formState, setFormState] = React.useState<Record<string, any>>({});

  // Create a placeholder schema if we only have the "tool_input_schema" string
  const schema: InputSchema = React.useMemo(() => {
    if (typeof tool.inputSchema === 'string') {
      // Default schema with a single text field
      return {
        type: "object",
        properties: {
          input: {
            type: "string",
            description: "Input for this tool"
          }
        },
        required: ["input"]
      };
    }
    return tool.inputSchema as InputSchema;
  }, [tool.inputSchema]);

  const handleInputChange = (key: string, value: any) => {
    setFormState(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formState);
  };

  return (
    <div className="bg-white rounded-lg shadow-lg border p-6 max-w-4xl w-full">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h2 className="text-xl font-bold">Test Tool: <span className="font-mono">{tool.name}</span></h2>
          <p className="text-gray-600">{tool.description}</p>
          <p className="text-sm text-gray-500 mt-1">Provider: {tool.mcp_info.server_name}</p>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-full hover:bg-gray-200"
        >
          <svg 
            xmlns="http://www.w3.org/2000/svg" 
            width="20" 
            height="20" 
            viewBox="0 0 24 24" 
            fill="none" 
            stroke="currentColor" 
            strokeWidth="2" 
            strokeLinecap="round" 
            strokeLinejoin="round"
          >
            <line x1="18" y1="6" x2="6" y2="18"></line>
            <line x1="6" y1="6" x2="18" y2="18"></line>
          </svg>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Form Section */}
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="font-medium mb-4">Input Parameters</h3>
          <form onSubmit={handleSubmit}>
            {typeof tool.inputSchema === 'string' ? (
              <div className="mb-4">
                <p className="text-xs text-gray-500 mb-1">This tool uses a dynamic input schema.</p>
                <div className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Input <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="text"
                    value={formState.input || ""}
                    onChange={(e) => handleInputChange("input", e.target.value)}
                    required
                    className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                  />
                </div>
              </div>
            ) : schema.properties === undefined ? (
              <p className="text-xs">None</p>
            ): (
              Object.entries(schema.properties).map(([key, prop]) => (
                <div key={key} className="mb-4">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    {key}{" "}
                    {schema.required?.includes(key) && (
                      <span className="text-red-500">*</span>
                    )}
                  </label>
                  {prop.description && (
                    <p className="text-xs text-gray-500 mb-1">{prop.description}</p>
                  )}
                  
                  {/* Render appropriate input based on type */}
                  {prop.type === "string" && (
                    <input
                      type="text"
                      value={formState[key] || ""}
                      onChange={(e) => handleInputChange(key, e.target.value)}
                      required={schema.required?.includes(key)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                    />
                  )}
                  
                  {prop.type === "number" && (
                    <input
                      type="number"
                      value={formState[key] || ""}
                      onChange={(e) => handleInputChange(key, parseFloat(e.target.value))}
                      required={schema.required?.includes(key)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                    />
                  )}
                  
                  {prop.type === "boolean" && (
                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        checked={formState[key] || false}
                        onChange={(e) => handleInputChange(key, e.target.checked)}
                        className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      />
                      <span className="ml-2 text-sm text-gray-600">Enable</span>
                    </div>
                  )}
                </div>
              ))
            )}
            
            <div className="mt-6">
              <Button
                type="submit"
                disabled={isLoading}
                className="w-full px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white"
              >
                {isLoading ? "Calling..." : "Call Tool"}
              </Button>
            </div>
          </form>
        </div>
        
        {/* Result Section */}
        <div className="bg-gray-50 p-4 rounded-lg overflow-auto max-h-[500px]">
          <h3 className="font-medium mb-4">Result</h3>
          
          {isLoading && (
            <div className="flex justify-center items-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-700"></div>
            </div>
          )}
          
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-md">
              <p className="font-medium">Error</p>
              <pre className="mt-2 text-xs overflow-auto whitespace-pre-wrap">{error.message}</pre>
            </div>
          )}
          
          {result && !isLoading && !error && (
            <div>
              {result.map((content: any, idx: number) => (
                <div key={idx} className="mb-4">
                  {content.type === "text" && (
                    <div className="bg-white border p-3 rounded-md">
                      <p className="whitespace-pre-wrap text-sm">{content.text}</p>
                    </div>
                  )}
                  
                  {content.type === "image" && content.url && (
                    <div className="bg-white border p-3 rounded-md">
                      <img src={content.url} alt="Tool result" className="max-w-full h-auto rounded" />
                    </div>
                  )}
                  
                  {content.type === "embedded_resource" && (
                    <div className="bg-white border p-3 rounded-md">
                      <p className="text-sm font-medium">Embedded Resource</p>
                      <p className="text-xs text-gray-500">Type: {content.resource_type}</p>
                      {content.url && (
                        <a 
                          href={content.url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-sm text-blue-600 hover:underline"
                        >
                          View Resource
                        </a>
                      )}
                    </div>
                  )}
                </div>
              ))}
              
              <div className="mt-2">
                <details className="text-xs">
                  <summary className="cursor-pointer text-gray-500 hover:text-gray-700">Raw JSON Response</summary>
                  <pre className="mt-2 bg-gray-100 p-2 rounded-md overflow-auto max-h-[300px]">
                    {JSON.stringify(result, null, 2)}
                  </pre>
                </details>
              </div>
            </div>
          )}
          
          {!result && !isLoading && !error && (
            <div className="text-center py-8 text-gray-500">
              <p>The result will appear here after you call the tool.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
} 