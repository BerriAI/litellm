import React, { useState } from "react";
import { Card, Text } from "@tremor/react";
import { Select } from "antd";
import { PlusIcon, TrashIcon, GripVerticalIcon } from "lucide-react";
import VariableTextArea from "../variable_textarea";
import { Message } from "./types";

const { Option } = Select;

interface PromptMessagesCardProps {
  messages: Message[];
  onAddMessage: () => void;
  onUpdateMessage: (index: number, field: "role" | "content", value: string) => void;
  onRemoveMessage: (index: number) => void;
  onMoveMessage: (fromIndex: number, toIndex: number) => void;
}

const PromptMessagesCard: React.FC<PromptMessagesCardProps> = ({
  messages,
  onAddMessage,
  onUpdateMessage,
  onRemoveMessage,
  onMoveMessage,
}) => {
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);

  const handleDragStart = (index: number) => {
    setDraggedIndex(index);
  };

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    setDragOverIndex(index);
  };

  const handleDrop = (e: React.DragEvent, dropIndex: number) => {
    e.preventDefault();
    if (draggedIndex !== null && draggedIndex !== dropIndex) {
      onMoveMessage(draggedIndex, dropIndex);
    }
    setDraggedIndex(null);
    setDragOverIndex(null);
  };

  const handleDragEnd = () => {
    setDraggedIndex(null);
    setDragOverIndex(null);
  };

  return (
    <Card className="p-3">
      <div className="mb-2">
        <Text className="text-sm font-medium">Prompt messages</Text>
        <Text className="text-gray-500 text-xs mt-1">
          Use <code className="bg-gray-100 px-1 rounded text-xs">{'{{variable}}'}</code> syntax for template variables
        </Text>
      </div>
      <div className="space-y-2">
        {messages.map((message, index) => (
          <div
            key={index}
            draggable
            onDragStart={() => handleDragStart(index)}
            onDragOver={(e) => handleDragOver(e, index)}
            onDrop={(e) => handleDrop(e, index)}
            onDragEnd={handleDragEnd}
            className={`border border-gray-300 rounded overflow-hidden bg-white transition-all ${
              draggedIndex === index ? "opacity-50" : ""
            } ${dragOverIndex === index && draggedIndex !== index ? "border-blue-500 border-2" : ""}`}
          >
            <div className="bg-gray-50 px-2 py-1.5 border-b border-gray-300 flex items-center justify-between">
              <Select
                value={message.role}
                onChange={(value) => onUpdateMessage(index, "role", value)}
                style={{ width: 100 }}
                size="small"
                bordered={false}
              >
                <Option value="user">User</Option>
                <Option value="assistant">Assistant</Option>
                <Option value="system">System</Option>
              </Select>
              <div className="flex items-center gap-1">
                {messages.length > 1 && (
                  <button
                    onClick={() => onRemoveMessage(index)}
                    className="text-gray-400 hover:text-red-500"
                  >
                    <TrashIcon size={14} />
                  </button>
                )}
                <div className="cursor-grab active:cursor-grabbing text-gray-400 hover:text-gray-600">
                  <GripVerticalIcon size={16} />
                </div>
              </div>
            </div>
            <div className="p-2">
              <VariableTextArea
                value={message.content}
                onChange={(value) => onUpdateMessage(index, "content", value)}
                rows={3}
                placeholder="Enter prompt content..."
              />
            </div>
          </div>
        ))}
      </div>
      <button
        onClick={onAddMessage}
        className="mt-2 text-xs text-blue-600 hover:text-blue-700 flex items-center"
      >
        <PlusIcon size={14} className="mr-1" />
        Add message
      </button>
    </Card>
  );
};

export default PromptMessagesCard;

