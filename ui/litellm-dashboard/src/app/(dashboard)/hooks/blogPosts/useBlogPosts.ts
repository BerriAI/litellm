import { getProxyBaseUrl } from "@/components/networking";
import { useQuery } from "@tanstack/react-query";

export interface BlogPost {
  title: string;
  description: string;
  date: string;
  url: string;
}

export interface BlogPostsResponse {
  posts: BlogPost[];
}

async function fetchBlogPosts(): Promise<BlogPostsResponse> {
  const baseUrl = getProxyBaseUrl();
  const response = await fetch(`${baseUrl}/public/litellm_blog_posts`);
  if (!response.ok) {
    throw new Error(`Failed to fetch blog posts: ${response.statusText}`);
  }
  return response.json();
}

export const useBlogPosts = () => {
  return useQuery<BlogPostsResponse>({
    queryKey: ["blogPosts"],
    queryFn: fetchBlogPosts,
    staleTime: 60 * 60 * 1000,
    retry: 1,
    retryDelay: 0,
  });
};
