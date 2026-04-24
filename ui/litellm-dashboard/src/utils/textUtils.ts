export const formatLabel = (text: string): string => {
  if (!text) {
    return text;
  }

  const withSpaces = text.replace(/_/g, " ");
  return withSpaces.replace(/\b\w/g, (char) => char.toUpperCase());
};

export function truncateString(str: string, maxLength: number) {
  return str.length > maxLength ? str.substring(0, maxLength) + "..." : str;
}

export const formItemValidateJSON = (_: any, value: string) => {
  if (!value) {
    return Promise.resolve();
  }
  try {
    JSON.parse(value);
    return Promise.resolve();
  } catch (error) {
    return Promise.reject("Please enter valid JSON");
  }
};

/**
 * React Hook Form-compatible JSON validator. Returns `true` when `value`
 * is empty or parses as valid JSON, otherwise the error message string.
 */
export const validateJsonValue = (value: unknown): true | string => {
  if (value === undefined || value === null || value === "") {
    return true;
  }
  try {
    JSON.parse(value as string);
    return true;
  } catch {
    return "Please enter valid JSON";
  }
};
