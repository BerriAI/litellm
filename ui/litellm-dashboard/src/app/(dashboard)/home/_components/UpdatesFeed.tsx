"use client";

import { ArrowUpRight, Rss } from "lucide-react";
import { useBlogPosts, type BlogPost } from "@/app/(dashboard)/hooks/blogPosts/useBlogPosts";
import { useDisableBlogPosts } from "@/app/(dashboard)/hooks/useDisableBlogPosts";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Panel } from "@/app/(dashboard)/usage/_components/components/overview/Primitives";

const formatDate = (isoDate: string): string =>
  new Date(`${isoDate}T00:00:00`).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });

export const BLOG_URL = "https://docs.litellm.ai/blog";

/** Latest posts from the LiteLLM blog RSS feed, via /public/litellm_blog_posts. */
export default function UpdatesFeed({ className }: { className?: string }) {
  const disabled = useDisableBlogPosts();
  const { data, isLoading, isError, refetch } = useBlogPosts();
  if (disabled) return null;

  return (
    <Panel
      icon={Rss}
      title="Updates"
      subtitle="From the LiteLLM blog"
      className={className}
      bodyClassName="pt-1"
      testId="home-updates"
      action={
        <Button variant="ghost" size="sm" render={<a href={BLOG_URL} target="_blank" rel="noopener noreferrer" />}>
          View all
          <ArrowUpRight className="size-3.5" />
        </Button>
      }
    >
      {isLoading && (
        <div className="flex flex-col gap-4 py-3">
          {[0, 1, 2].map((i) => (
            <div key={i} className="flex flex-col gap-2">
              <Skeleton className="h-4 w-4/5" />
              <Skeleton className="h-3 w-1/3" />
            </div>
          ))}
        </div>
      )}
      {isError && (
        <div className="flex items-center justify-between gap-2 py-3 text-sm">
          <span className="text-destructive">Failed to load posts</span>
          <Button variant="outline" size="sm" onClick={() => void refetch()}>
            Retry
          </Button>
        </div>
      )}
      {data && data.posts.length === 0 && <p className="py-3 text-sm text-muted-foreground">No posts yet</p>}
      {data && data.posts.length > 0 && (
        <ol className="divide-y divide-border/60">
          {data.posts.map((post: BlogPost) => (
            <li key={post.url}>
              <a
                href={post.url}
                target="_blank"
                rel="noopener noreferrer"
                className="group flex flex-col gap-1 py-3 outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
              >
                <span className="text-sm font-medium leading-5 text-foreground group-hover:underline">
                  {post.title}
                </span>
                <span className="text-xs text-muted-foreground">{formatDate(post.date)}</span>
                <span className="line-clamp-2 text-xs leading-4 text-muted-foreground">{post.description}</span>
              </a>
            </li>
          ))}
        </ol>
      )}
    </Panel>
  );
}
