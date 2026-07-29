import { useSearchParams } from "next/navigation";
import { useCallback } from "react";

import { navigateWithParams } from "../navigateWithParams";

export interface DetailParam {
  id: string | null;
  open: (id: string) => void;
  close: () => void;
}

export function useDetailParam(param: string): DetailParam {
  const searchParams = useSearchParams();

  const open = useCallback(
    (id: string) => {
      navigateWithParams((params) => {
        params.set(param, id);
      });
    },
    [param],
  );

  const close = useCallback(() => {
    navigateWithParams((params) => {
      params.delete(param);
    });
  }, [param]);

  return {
    id: searchParams?.get(param) ?? null,
    open,
    close,
  };
}
