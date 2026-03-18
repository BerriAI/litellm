import { ClearOutlined, CodeOutlined } from "@ant-design/icons";
import { Title, Button as TremorButton } from "@tremor/react";

interface ChatHeaderProps {
  simplified: boolean;
  onClearChat: () => void;
  onGetCode: () => void;
}

function ChatHeader({ simplified, onClearChat, onGetCode }: ChatHeaderProps) {
  return (
    <div className="p-4 border-b border-gray-200 flex justify-between items-center">
      <Title className="text-xl font-semibold mb-0">
        {simplified ? "Chat" : "Test Key"}
      </Title>
      <div className="flex gap-2">
        <TremorButton
          onClick={onClearChat}
          className="bg-gray-100 hover:bg-gray-200 text-gray-700 border-gray-300"
          icon={ClearOutlined}
        >
          Clear Chat
        </TremorButton>
        {!simplified && (
          <TremorButton
            onClick={onGetCode}
            className="bg-gray-100 hover:bg-gray-200 text-gray-700 border-gray-300"
            icon={CodeOutlined}
          >
            Get Code
          </TremorButton>
        )}
      </div>
    </div>
  );
}

export default ChatHeader;
