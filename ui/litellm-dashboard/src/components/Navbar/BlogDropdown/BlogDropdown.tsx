import {
  useBlogPosts,
  type BlogPost,
} from "@/app/(dashboard)/hooks/blogPosts/useBlogPosts";
import { useDisableBlogPosts } from "@/app/(dashboard)/hooks/useDisableBlogPosts";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Loader2 } from "lucide-react";
import React from "react";

function formatDate(dateStr: string): string {
  const date = new Date(dateStr + "T00:00:00");
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export const BlogDropdown: React.FC = () => {
  const disableBlogPosts = useDisableBlogPosts();
  const { data, isLoading, isError, refetch } = useBlogPosts();

  if (disableBlogPosts) {
    return null;
  }

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="ghost">Blog</Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-[420px] p-1">
        {isLoading ? (
          <div className="flex items-center justify-center py-4">
            <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />
          </div>
        ) : isError ? (
          <div className="flex items-center justify-between gap-2 p-3">
            <span className="text-destructive text-sm">
              Failed to load posts
            </span>
            <Button size="sm" variant="outline" onClick={() => refetch()}>
              Retry
            </Button>
          </div>
        ) : !data || data.posts.length === 0 ? (
          <div className="px-3 py-3 text-muted-foreground text-sm">
            No posts available
          </div>
        ) : (
          <>
            {data.posts.slice(0, 5).map((post: BlogPost) => (
              <DropdownMenuItem
                key={post.url}
                asChild
                className="cursor-pointer"
              >
                <a
                  href={post.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block w-full px-2 py-2"
                >
                  <h5 className="font-semibold text-sm mb-0.5">
                    {post.title}
                  </h5>
                  <div className="text-[11px] text-muted-foreground mb-0.5">
                    {formatDate(post.date)}
                  </div>
                  <p className="text-xs line-clamp-2 text-muted-foreground m-0">
                    {post.description}
                  </p>
                </a>
              </DropdownMenuItem>
            ))}
            <DropdownMenuSeparator />
            <DropdownMenuItem asChild>
              <a
                href="https://docs.litellm.ai/blog"
                target="_blank"
                rel="noopener noreferrer"
                className="block w-full px-2 py-2 text-sm"
              >
                View all posts
              </a>
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
};

export default BlogDropdown;
