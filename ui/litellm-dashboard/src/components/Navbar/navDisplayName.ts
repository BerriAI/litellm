/** Primary label for the navbar account control — avoids raw placeholder JWT/user IDs in the UI. */
export function navAccountDisplayName(userEmail: string | null, userId: string | null, fallback = "Account"): string {
  const email = userEmail?.trim();
  if (email) {
    return email;
  }
  const id = userId?.trim();
  if (!id) {
    return fallback;
  }
  if (/^default[_\s-]?user[_\s-]?id$/i.test(id)) {
    return fallback;
  }
  return id;
}
