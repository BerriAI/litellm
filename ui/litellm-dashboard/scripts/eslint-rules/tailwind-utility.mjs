const OPENERS = { "[": "]", "(": ")" };

export const utilityOf = (token) => {
  const closers = [];
  const lastTopLevelColon = [...token].reduce((found, ch, i) => {
    if (closers.length > 0 && ch === closers[closers.length - 1]) {
      closers.pop();
      return found;
    }
    if (ch in OPENERS) {
      closers.push(OPENERS[ch]);
      return found;
    }
    return ch === ":" && closers.length === 0 ? i : found;
  }, -1);
  return token
    .slice(lastTopLevelColon + 1)
    .replace(/^!/, "")
    .replace(/!$/, "");
};
