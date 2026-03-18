import { EndpointType } from "./mode_endpoint_mapping";

interface SuggestedPromptsProps {
  endpointType: string;
  onPromptClick: (prompt: string) => void;
}

const A2A_PROMPTS = [
  "What can you help me with?",
  "Tell me about yourself",
  "What tasks can you perform?",
];

const DEFAULT_PROMPTS = [
  "Write me a poem",
  "Explain quantum computing",
  "Draft a polite email requesting a meeting",
];

function SuggestedPrompts({ endpointType, onPromptClick }: SuggestedPromptsProps) {
  const prompts =
    endpointType === EndpointType.A2A_AGENTS ? A2A_PROMPTS : DEFAULT_PROMPTS;

  return (
    <div className="flex items-center gap-2 mb-3 overflow-x-auto">
      {prompts.map((prompt) => (
        <button
          key={prompt}
          type="button"
          className="shrink-0 rounded-full border border-gray-200 px-3 py-1 text-xs font-medium text-gray-600 transition-colors hover:bg-blue-50 hover:border-blue-300 hover:text-blue-600 cursor-pointer"
          onClick={() => onPromptClick(prompt)}
        >
          {prompt}
        </button>
      ))}
    </div>
  );
}

export default SuggestedPrompts;
