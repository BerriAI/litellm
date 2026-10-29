// QA probe for the stale base.sha check
export function mapEmptyStringToNull(input: string): string | null {
  if (input === "") {
    return null;
  }

  return input;
}
