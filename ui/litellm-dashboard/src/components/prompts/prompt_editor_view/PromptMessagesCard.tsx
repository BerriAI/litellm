import React, { useState } from "react";
import { Card } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { GripVertical, Plus, Trash } from "lucide-react";
import { cn } from "@/lib/utils";
import VariableTextArea from "../variable_textarea";
import { Message } from "./types";

interface PromptMessagesCardProps {
  messages: Message[];
  onAddMessage: () => void;
  onUpdateMessage: (
    index: number,
    field: "role" | "content",
    value: string,
  ) => void;
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
        <div className="text-sm font-medium">Prompt messages</div>
        <div className="text-muted-foreground text-xs mt-1">
          Use{" "}
          <code className="bg-muted px-1 rounded text-xs">
            {"{{variable}}"}
          </code>{" "}
          syntax for template variables
        </div>
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
            className={cn(
              "border border-border rounded overflow-hidden bg-background transition-all",
              draggedIndex === index && "opacity-50",
              dragOverIndex === index &&
                draggedIndex !== index &&
                "border-primary border-2",
            )}
          >
            <div className="bg-muted px-2 py-1.5 border-b border-border flex items-center justify-between">
              <Select
                value={message.role}
                onValueChange={(v) => onUpdateMessage(index, "role", v)}
              >
                <SelectTrigger className="w-[100px] h-7 border-0 bg-transparent shadow-none">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="user">User</SelectItem>
                  <SelectItem value="assistant">Assistant</SelectItem>
                  <SelectItem value="system">System</SelectItem>
                </SelectContent>
              </Select>
              <div className="flex items-center gap-1">
                {messages.length > 1 && (
                  <button
                    type="button"
                    onClick={() => onRemoveMessage(index)}
                    className="text-muted-foreground hover:text-destructive"
                    aria-label="Remove message"
                  >
                    <Trash size={14} />
                  </button>
                )}
                <div className="cursor-grab active:cursor-grabbing text-muted-foreground hover:text-foreground">
                  <GripVertical size={16} />
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
        type="button"
        onClick={onAddMessage}
        className="mt-2 text-xs text-primary hover:underline flex items-center"
      >
        <Plus size={14} className="mr-1" />
        Add message
      </button>
    </Card>
  );
};

export default PromptMessagesCard;
