import type { BlogPost } from "@/app/(dashboard)/hooks/blogPosts/useBlogPosts";

export const DAY_ZERO_TAG = "day 0 support";

export const WHATS_NEW_MAX = 3;

export const latestLaunches = (posts: readonly BlogPost[]): readonly BlogPost[] =>
  posts
    .filter((post) => post.tags.includes(DAY_ZERO_TAG))
    .sort((a, b) => b.date.localeCompare(a.date))
    .slice(0, WHATS_NEW_MAX);

export const formatPublishedOn = (isoDate: string): string =>
  new Date(`${isoDate}T00:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric" });
