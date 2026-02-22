import { useDisableShowBlog } from "@/app/(dashboard)/hooks/useDisableShowBlog";
import { getProxyBaseUrl } from "@/components/networking";
import {
  DownOutlined,
  LoadingOutlined,
  ReadOutlined,
} from "@ant-design/icons";
import { useQuery } from "@tanstack/react-query";
import { Button, Dropdown, Space, Typography } from "antd";
import React from "react";

const { Text } = Typography;

interface BlogPost {
  title: string;
  description: string;
  date: string;
  url: string;
}

interface BlogPostsResponse {
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

function formatDate(dateStr: string): string {
  const date = new Date(dateStr + "T00:00:00");
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export const BlogDropdown: React.FC = () => {
  const disableShowBlog = useDisableShowBlog();

  const { data, isLoading, isError, refetch } = useQuery<BlogPostsResponse>({
    queryKey: ["blogPosts"],
    queryFn: fetchBlogPosts,
    staleTime: 60 * 60 * 1000, // 1 hour — matches server-side TTL
  });

  if (disableShowBlog) {
    return null;
  }

  const dropdownContent = () => {
    if (isError) {
      return (
        <div style={{ padding: "12px 16px", minWidth: 200 }}>
          <Text type="danger" style={{ display: "block", marginBottom: 8 }}>
            Failed to load blog posts
          </Text>
          <Button size="small" onClick={() => refetch()}>
            Retry
          </Button>
        </div>
      );
    }

    if (!data || data.posts.length === 0) {
      return (
        <div style={{ padding: "12px 16px" }}>
          <Text type="secondary">No posts available</Text>
        </div>
      );
    }

    return (
      <div style={{ minWidth: 280, maxWidth: 360 }}>
        {data.posts.slice(0, 5).map((post, index) => (
          <a
            key={index}
            href={post.url}
            target="_blank"
            rel="noopener noreferrer"
            style={{ display: "block", textDecoration: "none", color: "inherit" }}
          >
            <div
              style={{ padding: "10px 16px" }}
              className="hover:bg-gray-50 transition-colors cursor-pointer"
            >
              <div
                style={{
                  fontWeight: 500,
                  fontSize: 13,
                  marginBottom: 2,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {post.title}
              </div>
              <div style={{ fontSize: 11, color: "#8c8c8c", marginBottom: 3 }}>
                {formatDate(post.date)}
              </div>
              <div
                style={{
                  fontSize: 12,
                  color: "#595959",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {post.description}
              </div>
            </div>
          </a>
        ))}
      </div>
    );
  };

  return (
    <Dropdown
      popupRender={dropdownContent}
      trigger={["click"]}
      placement="bottomRight"
    >
      <Button
        type="text"
        className="text-sm text-gray-600 hover:text-gray-900 transition-colors"
        icon={<ReadOutlined />}
      >
        <Space>
          Blog
          {isLoading ? <LoadingOutlined /> : <DownOutlined />}
        </Space>
      </Button>
    </Dropdown>
  );
};

export default BlogDropdown;
