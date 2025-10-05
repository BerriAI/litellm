import { useEffect, useMemo, useState } from "react";
import { debounce } from "lodash";
import { searchUserOptionsByEmail } from "@/app/(console)/virtual-keys/components/CreateKeyModal/networking";
import useAuthorized from "@/app/(console)/hooks/useAuthorized";

type User = { user_id: string; user_email: string; role?: string };
export interface UserOption {
  label: string;
  value: string;
  user: User;
}

export const useUserSearch = (delay = 300) => {
  const { accessToken } = useAuthorized();
  const [options, setOptions] = useState<UserOption[]>([]);
  const [loading, setLoading] = useState(false);

  const doSearch = async (raw: string) => {
    const q = raw?.trim();
    if (!q || !accessToken) {
      setOptions([]);
      setLoading(false);
      return;
    }
    setLoading(true);
    try {
      const res = await searchUserOptionsByEmail(accessToken, q);
      setOptions(res || []);
    } finally {
      setLoading(false);
    }
  };

  const onSearch = useMemo(() => debounce(doSearch, delay), [accessToken, delay]);

  useEffect(() => {
    return () => onSearch.cancel();
  }, [onSearch]);

  return { options, loading, onSearch };
};
