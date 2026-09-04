import { migratedHref } from "@/utils/migratedPages";

export interface TabRoutes<Slug extends string> {
  baseSegment: string;
  slugs: readonly Slug[];
  tabHref: (slug: string) => string;
  slugFromPathname: (pathname: string) => string;
}

export function createTabRoutes<Slug extends string>(baseSegment: string, slugs: readonly Slug[]): TabRoutes<Slug> {
  const tabHref = (slug: string): string => {
    const base = migratedHref(baseSegment);
    return slug ? `${base}/${slug}/` : `${base}/`;
  };

  const slugFromPathname = (pathname: string): string => {
    const parts = pathname.split("/").filter(Boolean);
    const idx = parts.indexOf(baseSegment);
    if (idx === -1) {
      return "";
    }
    return parts[idx + 1] ?? "";
  };

  return { baseSegment, slugs, tabHref, slugFromPathname };
}
