const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === "object" && value !== null;

export const streamErrorMessage = (frame: unknown): string | null => {
  if (!isRecord(frame) || !frame.error) {
    return null;
  }
  const { error } = frame;
  if (typeof error === "string") {
    return error;
  }
  if (isRecord(error) && typeof error.message === "string") {
    return error.message;
  }
  return JSON.stringify(error);
};
