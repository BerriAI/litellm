export function truncateString(str: string, maxLength: number) {
  return str.length > maxLength ? str.substring(0, maxLength) + "..." : str;
}
