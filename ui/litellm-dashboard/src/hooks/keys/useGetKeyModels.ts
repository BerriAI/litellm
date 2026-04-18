import { useQuery } from '@tanstack/react-query';
import { keyModelCall } from '@/components/networking';
import useAuthorized from '@/app/(dashboard)/hooks/useAuthorized';

export const UseGetKeyModels = (key_id: string) => {
  const { accessToken } = useAuthorized();
  
  return useQuery({
    queryKey: ['keyModels', key_id],
    queryFn: () => {
        if (!accessToken) throw new Error("Access Token required");
        return keyModelCall(accessToken, key_id);
    },
  });
};
