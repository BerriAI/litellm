import React from "react";
import { MessageSquareIcon } from "lucide-react";

const ConversationPanel: React.FC = () => {
  return (
    <div className="flex-1 bg-white flex flex-col">
      <div className="flex-1 flex items-center justify-center text-gray-400">
        <div className="text-center">
          <div className="w-12 h-12 mx-auto mb-3 bg-gray-100 rounded-full flex items-center justify-center">
            <MessageSquareIcon size={24} className="text-gray-400" />
          </div>
          <p className="text-sm">Your conversation will appear here</p>
          <p className="text-xs text-gray-500 mt-2">Save the prompt to test it</p>
        </div>
      </div>
    </div>
  );
};

export default ConversationPanel;

