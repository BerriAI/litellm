"use client";

import Usage from "@/components/usage";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useState } from "react";

const OldUsagePage = () => {
  const { accessToken, token, userRole, userId, premiumUser } = useAuthorized();
  const [keys, setKeys] = useState<null | any[]>([]);

  return (
    <Usage
      accessToken={accessToken}
      token={token}
      userRole={userRole}
      userID={userId}
      keys={keys}
      premiumUser={premiumUser}
    />
  );
};

export default OldUsagePage;
