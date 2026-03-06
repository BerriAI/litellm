"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import ChatPage from "@/components/chat/ChatPage";

const ChatPageRoute = () => {
  const { accessToken, userRole, userId, userEmail } = useAuthorized();

  return (
    <ChatPage
      accessToken={accessToken ?? ""}
      userRole={userRole ?? ""}
      userId={userId ?? ""}
      userEmail={userEmail ?? ""}
    />
  );
};

export default ChatPageRoute;
