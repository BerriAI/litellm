export const queryState = (isPending: boolean, isEnabled = true) => ({
  isPending,
  isEnabled,
  isLoading: isEnabled && isPending,
});
