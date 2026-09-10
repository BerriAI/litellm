import { persistFile } from './api';

export const createFileSaver = (onError: (target: string) => void) => {
  const timers = new Map<string, number>();

  return (target: string, source: string) => {
    const pendingTimer = timers.get(target);
    if (pendingTimer !== undefined) {
      window.clearTimeout(pendingTimer);
    }
    timers.set(target, window.setTimeout(async () => {
      timers.delete(target);
      try {
        await persistFile(target, source);
      } catch {
        onError(target);
      }
    }, 350));
  };
};
