import { useDisableBlogPosts } from "@/app/(dashboard)/hooks/useDisableBlogPosts";
import { useBlogPosts, type BlogPost } from "@/app/(dashboard)/hooks/blogPosts/useBlogPosts";
import { NAV_PRODUCT_LINK_CLASS } from "@/components/Navbar/navProductLinkClass";
import { DownOutlined, LoadingOutlined } from "@ant-design/icons";
import { Button, Dropdown, Space, Typography } from "antd";
import type { MenuProps } from "antd";
import React from "react";
import { useTranslation } from "react-i18next";

const { Text, Title, Paragraph } = Typography;

function formatDate(dateStr: string, locale: string): string {
  const date = new Date(dateStr + "T00:00:00");
  return date.toLocaleDateString(locale, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export const BlogDropdown: React.FC = () => {
  const { t, i18n } = useTranslation("common");
  const disableBlogPosts = useDisableBlogPosts();
  const locale = i18n.resolvedLanguage === "ru" ? "ru-RU" : "en-US";

  const { data, isLoading, isError, refetch } = useBlogPosts();

  if (disableBlogPosts) {
    return null;
  }

  let items: MenuProps["items"];

  if (isLoading) {
    items = [{ key: "loading", label: <LoadingOutlined />, disabled: true }];
  } else if (isError) {
    items = [
      {
        key: "error",
        label: (
          <Space>
            <Text type="danger">{t("nav.blog.failed")}</Text>
            <Button size="small" onClick={() => refetch()}>
              {t("nav.blog.retry")}
            </Button>
          </Space>
        ),
        disabled: true,
      },
    ];
  } else if (!data || data.posts.length === 0) {
    items = [{ key: "empty", label: <Text type="secondary">{t("nav.blog.empty")}</Text>, disabled: true }];
  } else {
    items = [
      ...data.posts.slice(0, 5).map((post: BlogPost) => ({
        key: post.url,
        label: (
          <a href={post.url} target="_blank" rel="noopener noreferrer" style={{ display: "block", width: 380 }}>
            <Title level={5} style={{ marginBottom: 2 }}>
              {post.title}
            </Title>
            <Text type="secondary" style={{ fontSize: 11 }}>
              {formatDate(post.date, locale)}
            </Text>
            <Paragraph ellipsis={{ rows: 2 }}>{post.description}</Paragraph>
          </a>
        ),
      })),
      { type: "divider" as const },
      {
        key: "view-all",
        label: (
          <a href="https://docs.litellm.ai/blog" target="_blank" rel="noopener noreferrer">
            {t("nav.blog.viewAll")}
          </a>
        ),
      },
    ];
  }

  // Blog opens a post list; Docs is a single outbound link — navbar adds a layout-only chevron there for alignment.
  return (
    <Dropdown menu={{ items }} trigger={["hover"]} placement="bottomRight">
      <Button type="text" className={`${NAV_PRODUCT_LINK_CLASS} border-0! bg-transparent!`}>
        {t("nav.blog.title")}
        <DownOutlined className="text-[10px] text-gray-500" aria-hidden />
      </Button>
    </Dropdown>
  );
};

export default BlogDropdown;
