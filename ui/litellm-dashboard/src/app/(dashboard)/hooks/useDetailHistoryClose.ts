import { useCallback, useEffect, useRef } from "react";

export interface DetailHistoryClose {
  markOpened: () => void;
  close: () => void;
}

export function useDetailHistoryClose(clearParams: () => void): DetailHistoryClose {
  const pushedCountRef = useRef(0);
  const suppressPopRef = useRef(false);

  useEffect(() => {
    const onPop = () => {
      if (suppressPopRef.current) {
        suppressPopRef.current = false;
        return;
      }
      if (pushedCountRef.current > 0) {
        pushedCountRef.current -= 1;
      }
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const markOpened = useCallback(() => {
    pushedCountRef.current += 1;
  }, []);

  const close = useCallback(() => {
    if (pushedCountRef.current > 0) {
      pushedCountRef.current -= 1;
      suppressPopRef.current = true;
      window.history.back();
      return;
    }
    clearParams();
  }, [clearParams]);

  return { markOpened, close };
}
