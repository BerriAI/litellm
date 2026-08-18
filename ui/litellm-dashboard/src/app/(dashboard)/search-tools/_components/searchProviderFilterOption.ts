interface SearchProviderOption {
  value?: string | number | null;
  title?: string;
}

// antd's default filter stringifies option children, which are React elements here, so it matches nothing
export const searchProviderFilterOption = (input: string, option?: SearchProviderOption): boolean => {
  const needle = input.trim().toLowerCase();
  if (!needle) return true;
  return [option?.value, option?.title].some((part) => typeof part === "string" && part.toLowerCase().includes(needle));
};
