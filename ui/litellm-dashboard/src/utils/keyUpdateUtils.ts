export function mapEmptyStringToNull(input: string): string | null {
  if (input === "") {
    return null;
  }

  return input;
}
