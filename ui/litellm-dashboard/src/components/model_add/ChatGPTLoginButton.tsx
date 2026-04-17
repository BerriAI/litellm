import React from "react";

import {
  chatgptOauthCancelCall,
  chatgptOauthStartCall,
  chatgptOauthStatusCall,
} from "@/components/networking";
import OAuthDeviceLoginButton from "./OAuthDeviceLoginButton";

interface ChatGPTLoginButtonProps {
  credentialName?: string;
  onSuccess: () => void;
}

const ChatGPTLoginButton: React.FC<ChatGPTLoginButtonProps> = (props) => (
  <OAuthDeviceLoginButton
    providerLabel="ChatGPT"
    startCall={chatgptOauthStartCall}
    statusCall={chatgptOauthStatusCall}
    cancelCall={chatgptOauthCancelCall}
    {...props}
  />
);

export default ChatGPTLoginButton;
