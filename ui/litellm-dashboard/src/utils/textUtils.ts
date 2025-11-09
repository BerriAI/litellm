export const formatLabel = (text: string): string => {
  if (!text) {
    return text;
  }

  const withSpaces = text.replace(/_/g, " ");
  return withSpaces.replace(/\b\w/g, (char) => char.toUpperCase());
};
