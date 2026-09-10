export const createFileSaver = (
  persist: (target: string, source: string) => Promise<unknown>,
) => {
  const timers = new Map<string, number>();

  return (target: string, source: string) => {
    const pendingTimer = timers.get(target);
    if (pendingTimer !== undefined) {
      window.clearTimeout(pendingTimer);
    }
    timers.set(target, window.setTimeout(() => {
      timers.delete(target);
      void persist(target, source).catch(() => undefined);
    }, 350));
  };
};
