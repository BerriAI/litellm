import ChatUI from "@/components/playground/chat_ui/ChatUI";
import CompareUI from "@/components/playground/compareUI/CompareUI";
import { TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useState } from "react";

export interface PlaygroundPageProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  disabledPersonalKeyCreation: boolean;
  proxySettings?: {
    PROXY_BASE_URL?: string;
    LITELLM_UI_API_DOC_BASE_URL?: string | null;
  };
}

export function PlaygroundPage({
  accessToken,
  token,
  userRole,
  userID,
  disabledPersonalKeyCreation,
  proxySettings,
}: PlaygroundPageProps) {
  return (
    <TabGroup className="h-full w-full">
      <TabList className="mb-0">
        <Tab>Chat</Tab>
        <Tab>Compare</Tab>
      </TabList>
      <TabPanels className="h-full">
        <TabPanel className="h-full">
          <ChatUI
            accessToken={accessToken}
            token={token}
            userRole={userRole}
            userID={userID}
            disabledPersonalKeyCreation={disabledPersonalKeyCreation}
            proxySettings={proxySettings}
          />
        </TabPanel>
        <TabPanel className="h-full">
          <CompareUI accessToken={accessToken} disabledPersonalKeyCreation={disabledPersonalKeyCreation} />
        </TabPanel>
      </TabPanels>
    </TabGroup>
  );
}

// Default export for Next.js page routing (uses useAuthorized hook)
export default function PlaygroundPageWithAuth() {
  const { accessToken, token, userRole, userId, disabledPersonalKeyCreation } = useAuthorized();
  const [proxySettings, setProxySettings] = useState<PlaygroundPageProps["proxySettings"]>(undefined);

  return (
    <PlaygroundPage
      accessToken={accessToken}
      token={token}
      userRole={userRole}
      userID={userId}
      disabledPersonalKeyCreation={disabledPersonalKeyCreation}
      proxySettings={proxySettings}
    />
  );
}
