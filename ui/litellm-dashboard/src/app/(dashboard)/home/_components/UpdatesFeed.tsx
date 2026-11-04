"use client";

import { ArrowUpRight, Newspaper } from "lucide-react";
import { useBlogPosts, type BlogPost } from "@/app/(dashboard)/hooks/blogPosts/useBlogPosts";
import { useDisableBlogPosts } from "@/app/(dashboard)/hooks/useDisableBlogPosts";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";

export const BLOG_URL = "https://docs.litellm.ai/blog";
const MAX_ROWS = 6;
const DAY_MS = 86_400_000;

const relativeTime = new Intl.RelativeTimeFormat("en", { numeric: "auto" });

export const formatRelativeDate = (isoDate: string, now: Date = new Date()): string => {
  const published = new Date(`${isoDate}T00:00:00`);
  if (Number.isNaN(published.getTime())) return isoDate;
  const days = Math.max(0, Math.round((now.getTime() - published.getTime()) / DAY_MS));
  if (days < 7) return relativeTime.format(-days, "day");
  if (days < 30) return relativeTime.format(-Math.round(days / 7), "week");
  if (days < 365) return relativeTime.format(-Math.round(days / 30), "month");
  return relativeTime.format(-Math.round(days / 365), "year");
};

const Row = ({ children }: { children: React.ReactNode }) => (
  <li className="flex gap-3 py-3">
    <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full border border-border bg-background text-muted-foreground">
      <Newspaper className="size-3.5" />
    </span>
    <div className="flex min-w-0 flex-1 flex-col gap-0.5">{children}</div>
  </li>
);

/** Latest posts from the LiteLLM blog RSS feed, via /public/litellm_blog_posts, one row per post. */
export default function UpdatesFeed({ className }: { className?: string }) {
  const disabled = useDisableBlogPosts();
  const { data, isLoading, isError, refetch } = useBlogPosts();
  if (disabled) return null;

  const posts = data?.posts.slice(0, MAX_ROWS) ?? [];

  return (
    <section aria-labelledby="home-updates-title" data-testid="home-updates" className={className}>
      <div className="flex items-center justify-between gap-2">
        <h2 id="home-updates-title" className="text-lg font-semibold">
          Updates
        </h2>
        <Button variant="ghost" size="sm" render={<a href={BLOG_URL} target="_blank" rel="noopener noreferrer" />}>
          View all
          <ArrowUpRight className="size-3.5" />
        </Button>
      </div>
      {isLoading && (
        <ul className="m-0 list-none p-0">
          {[0, 1, 2, 3].map((i) => (
            <Row key={i}>
              <Skeleton className="h-3 w-16" />
              <Skeleton className="h-4 w-3/5" />
              <Skeleton className="h-3 w-full" />
            </Row>
          ))}
        </ul>
      )}
      {isError && (
        <div className="flex items-center justify-between gap-2 py-3 text-sm">
          <span className="text-destructive">Failed to load posts</span>
          <Button variant="outline" size="sm" onClick={() => void refetch()}>
            Retry
          </Button>
        </div>
      )}
      {data && posts.length === 0 && <p className="py-3 text-sm text-muted-foreground">No posts yet</p>}
      {posts.length > 0 && (
        <ol className="m-0 list-none p-0">
          {posts.map((post: BlogPost) => (
            <Row key={post.url}>
              <span className="text-xs text-muted-foreground">{formatRelativeDate(post.date)}</span>
              <a
                href={post.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm font-semibold leading-5 text-foreground outline-none hover:underline focus-visible:ring-2 focus-visible:ring-ring/50"
              >
                {post.title}
              </a>
              <span className="line-clamp-2 text-sm leading-5 text-muted-foreground">{post.description}</span>
            </Row>
          ))}
        </ol>
      )}
    </section>
  );
}
