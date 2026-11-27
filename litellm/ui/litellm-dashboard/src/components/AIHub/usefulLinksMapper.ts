import type { UsefulLink } from "./usefulLinksSchema";

export const toUsefulLinksRequest = (links: readonly Pick<UsefulLink, "displayName" | "url">[]) => {
  const useful_links: Record<string, { url: unknown; index: number }> = {};
  links.forEach(({ displayName, url }, index) => {
    useful_links[displayName] = { url, index };
  });
  return { useful_links };
};
