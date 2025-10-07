import { useEffect, useState } from "react";
import { getUserModelNames } from "@/app/(dashboard)/virtual-keys/components/CreateKeyModal/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export const useUserModels = () => {
  const { userId: userID, userRole, accessToken } = useAuthorized();
  const [userModels, setUserModels] = useState<string[]>([]);

  useEffect(() => {
    if (!userID || !userRole || !accessToken) {
      setUserModels([]);
      return;
    }
    (async () => {
      const modelNames = await getUserModelNames(userID, userRole, accessToken);
      setUserModels(modelNames || []);
    })();
  }, [userID, userRole, accessToken]);

  return userModels;
};
