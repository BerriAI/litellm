const RELEASE_VERSION_PATTERN = /^v?(\d+)\.(\d+)\.(\d+)/;

export const parseReleaseVersion = (version: string): readonly [number, number, number] | null => {
  const match = RELEASE_VERSION_PATTERN.exec(version.trim());
  if (!match) {
    return null;
  }
  return [Number(match[1]), Number(match[2]), Number(match[3])];
};

export const isNewerVersion = (current: string, latest: string): boolean => {
  const currentParts = parseReleaseVersion(current);
  const latestParts = parseReleaseVersion(latest);
  if (!currentParts || !latestParts) {
    return false;
  }
  for (let i = 0; i < 3; i += 1) {
    if (latestParts[i] !== currentParts[i]) {
      return latestParts[i] > currentParts[i];
    }
  }
  return false;
};
