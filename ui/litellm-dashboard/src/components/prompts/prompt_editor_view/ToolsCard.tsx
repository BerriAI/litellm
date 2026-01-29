import React from "react";
import { Card, Text } from "@tremor/react";
import { PlusIcon, TrashIcon } from "lucide-react";
import { Tool } from "./types";

interface ToolsCardProps {
  tools: Tool[];
  onAddTool: () => void;
  onEditTool: (index: number) => void;
  onRemoveTool: (index: number) => void;
}

const ToolsCard: React.FC<ToolsCardProps> = ({
  tools,
  onAddTool,
  onEditTool,
  onRemoveTool,
}) => {
  return (
    <Card className="p-3">
      <div className="flex items-center justify-between mb-2">
        <Text className="text-sm font-medium">Tools</Text>
        <button
          onClick={onAddTool}
          className="text-xs text-blue-600 hover:text-blue-700 flex items-center"
        >
          <PlusIcon size={14} className="mr-1" />
          Add
        </button>
      </div>
      {tools.length === 0 ? (
        <Text className="text-gray-500 text-xs">No tools added</Text>
      ) : (
        <div className="space-y-2">
          {tools.map((tool, index) => (
            <div
              key={index}
              className="flex items-center justify-between p-2 bg-gray-50 border border-gray-200 rounded"
            >
              <div className="flex-1 min-w-0">
                <div className="font-medium text-xs truncate">{tool.name}</div>
                <div className="text-xs text-gray-500 truncate">{tool.description}</div>
              </div>
              <div className="flex items-center space-x-1 ml-2">
                <button
                  onClick={() => onEditTool(index)}
                  className="text-xs text-blue-600 hover:text-blue-700"
                >
                  Edit
                </button>
                <button
                  onClick={() => onRemoveTool(index)}
                  className="text-gray-400 hover:text-red-500"
                >
                  <TrashIcon size={14} />
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
};

export default ToolsCard;

