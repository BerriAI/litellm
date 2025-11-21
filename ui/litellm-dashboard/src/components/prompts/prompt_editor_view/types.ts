export interface Message {
  role: string;
  content: string;
}

export interface Tool {
  name: string;
  description: string;
  json: string;
}

export interface PromptType {
  name: string;
  model: string;
  config: {
    temperature?: number;
    max_tokens?: number;
    top_p?: number;
  };
  tools: Tool[];
  developerMessage: string;
  messages: Message[];
}

export interface PromptEditorViewProps {
  onClose: () => void;
  onSuccess: () => void;
  accessToken: string | null;
}

