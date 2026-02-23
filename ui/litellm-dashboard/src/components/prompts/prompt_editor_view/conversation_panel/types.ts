import { TokenUsage } from "../../../playground/chat_ui/ResponseMetrics";

export interface Message {
  role: string;
  content: string;
  model?: string;
  timeToFirstToken?: number;
  totalLatency?: number;
  usage?: TokenUsage;
}

export interface ConversationPanelProps {
  prompt: any;
  accessToken: string | null;
}

