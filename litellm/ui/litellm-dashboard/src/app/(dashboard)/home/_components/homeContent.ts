import type { BlogPost } from "@/app/(dashboard)/hooks/blogPosts/useBlogPosts";
import i18n from "@/i18n/i18n";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";

export const DAY_ZERO_TAG = "day 0 support";

export const WHATS_NEW_MAX = 3;

const CARD_TITLE_TAG = "card_title:";

const CARD_SUMMARY_TAG = "card_summary:";

export const latestLaunches = (posts: readonly BlogPost[]): readonly BlogPost[] =>
  posts
    .filter((post) => post.tags.includes(DAY_ZERO_TAG))
    .sort((a, b) => b.date.localeCompare(a.date))
    .slice(0, WHATS_NEW_MAX);

const taggedValue = (tags: readonly string[], prefix: string): string | undefined => {
  const value = tags
    .find((tag) => tag.startsWith(prefix))
    ?.slice(prefix.length)
    .trim();
  return value ? value : undefined;
};

export const firstSentence = (text: string): string => /^.*?[.!?](?=\s|$)/.exec(text)?.[0] ?? text;

export const cardTitle = (post: BlogPost): string => taggedValue(post.tags, CARD_TITLE_TAG) ?? post.title;

export const cardSummary = (post: BlogPost): string =>
  taggedValue(post.tags, CARD_SUMMARY_TAG) ?? firstSentence(post.description);

export const cardProvider = (post: BlogPost): string | undefined =>
  post.tags.find((tag) => getProviderLogoAndName(tag).logo !== "");

export const formatPublishedOn = (isoDate: string, language = i18n.language): string =>
  new Date(`${isoDate}T00:00:00`).toLocaleDateString(language, { month: "short", day: "numeric" });
