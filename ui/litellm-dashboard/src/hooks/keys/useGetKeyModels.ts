import { useQuery } from '@tanstack/react-query';
import { fetchKeyModelCall } from '@/components/networking';
import useAuthorized from '@/app/(dashboard)/hooks/useAuthorized';

export const UseGetKeyModels = (key_id: string) => {
  const { accessToken } = useAuthorized();
  
  return useQuery({
    queryKey: ['keyModels', key_id],
    queryFn: () => {
        if (!accessToken) throw new Error("Access Token required");
        return fetchKeyModelCall(accessToken, key_id);
    },
  });
};
