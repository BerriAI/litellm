export const durationOrNull = (value: string | number | null | undefined): string | null =>
  typeof value === "string" && value.trim() !== "" ? value.trim() : null;
