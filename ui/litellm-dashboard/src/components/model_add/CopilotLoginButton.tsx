import React from "react";

import {
  copilotOauthCancelCall,
  copilotOauthStartCall,
  copilotOauthStatusCall,
} from "@/components/networking";
import OAuthDeviceLoginButton from "./OAuthDeviceLoginButton";

interface CopilotLoginButtonProps {
  credentialName?: string;
  onSuccess: () => void;
}

const CopilotLoginButton: React.FC<CopilotLoginButtonProps> = (props) => (
  <OAuthDeviceLoginButton
    providerLabel="GitHub Copilot"
    startCall={copilotOauthStartCall}
    statusCall={copilotOauthStatusCall}
    cancelCall={copilotOauthCancelCall}
    {...props}
  />
);

export default CopilotLoginButton;
