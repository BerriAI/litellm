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
