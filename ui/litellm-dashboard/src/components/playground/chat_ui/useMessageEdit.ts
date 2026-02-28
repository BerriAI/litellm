import { useState } from "react";
import { MessageType } from "./types";

export const useMessageEdit = (
  chatHistory: MessageType[],
  setChatHistory: (history: MessageType[]) => void,
  onRetry: (content: string) => void
) => {
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editingContent, setEditingContent] = useState<string>("");

  const extractTextContent = (message: MessageType): string => {
    if (typeof message.content === "string") return message.content;
    if (Array.isArray(message.content)) {
      const textParts = message.content.filter(
        (part: any) => part.type === "text" || part.type === "input_text"
      );
      return textParts.map((part: any) => part.text).join("\n");
    }
    return "";
  };

  const startEdit = (index: number) => {
    const message = chatHistory[index];
    if (message.role !== "user") return;
    setEditingIndex(index);
    setEditingContent(extractTextContent(message));
  };

  const saveEdit = () => {
    if (editingIndex === null) return;
    
    const updatedHistory = [...chatHistory];
    updatedHistory[editingIndex] = {
      ...updatedHistory[editingIndex],
      content: editingContent,
    };
    
    setChatHistory(updatedHistory.slice(0, editingIndex + 1));
    setEditingIndex(null);
    setEditingContent("");
    
    setTimeout(() => onRetry(editingContent), 100);
  };

  const cancelEdit = () => {
    setEditingIndex(null);
    setEditingContent("");
  };

  const retry = (index: number) => {
    const message = chatHistory[index];
    if (message.role !== "user") return;
    
    const content = extractTextContent(message);
    setChatHistory(chatHistory.slice(0, index));
    
    setTimeout(() => onRetry(content), 100);
  };

  return {
    editingIndex,
    editingContent,
    setEditingContent,
    startEdit,
    saveEdit,
    cancelEdit,
    retry,
  };
};
