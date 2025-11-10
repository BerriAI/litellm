"use client";

import ChatUI from "@/components/chat_ui/ChatUI";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const TestKeyPage = () => {
  const { token, accessToken, userRole, userId, disabledPersonalKeyCreation } = useAuthorized();

  return (
    <ChatUI
      accessToken={accessToken}
      token={token}
      userRole={userRole}
      userID={userId}
      disabledPersonalKeyCreation={disabledPersonalKeyCreation}
    />
  );
};

export default TestKeyPage;
