import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { I18nextProvider } from "react-i18next";
import { createDashboardI18n } from "@/i18n";
import { AUTO_ROUTER_DOCS_URL, NotificationsBellView } from "./NotificationsBell";

describe("localized announcements", () => {
  beforeEach(() => localStorage.clear());
  afterEach(() => localStorage.clear());

  it("translates releases and unread counts while preserving independent read state and links", async () => {
    const user = userEvent.setup();
    const instance = createDashboardI18n("zh-CN");
    const releaseUrl = "https://github.com/BerriAI/litellm/releases/tag/v1.103.0";
    render(
      <I18nextProvider i18n={instance}>
        <NotificationsBellView
          currentVersion="1.102.0"
          latestRelease={{
            version: "1.103.0",
            new_features: 1,
            bug_fixes: 2,
            other_updates: 3,
            release_url: releaseUrl,
          }}
        />
      </I18nextProvider>,
    );
    await user.click(screen.getByRole("button", { name: "通知，2 条未读" }));
    const release = screen.getByRole("article", { name: "LiteLLM v1.103.0 已发布" });
    expect(release).toHaveTextContent("1 项新功能、2 项修复和3 项其他更新。当前版本为 v1.102.0");
    expect(within(release).getByRole("link", { name: "查看发行说明" })).toHaveAttribute("href", releaseUrl);
    expect(screen.getByRole("link", { name: "查看文档" })).toHaveAttribute("href", AUTO_ROUTER_DOCS_URL);
    await user.click(within(release).getByRole("button", { name: "标记为已读" }));
    expect(screen.getByRole("button", { name: "通知，1 条未读" })).toBeInTheDocument();
    expect(localStorage.getItem("litellm:upgradeBannerDismissed:1.103.0")).toBe("true");
    expect(localStorage.getItem("litellmHideAutoRouterAnnouncement")).toBeNull();

    await act(async () => {
      await instance.changeLanguage("en");
    });
    expect(screen.getByRole("button", { name: "Notifications, 1 unread" })).toBeInTheDocument();
    expect(screen.getByRole("article", { name: "LiteLLM v1.103.0 is available" })).toHaveTextContent(
      "1 new feature, 2 fixes, and 3 other updates. Your current version is v1.102.0",
    );
    await user.click(screen.getByRole("button", { name: "Mark as read" }));
    expect(screen.getByRole("button", { name: "Notifications", exact: true })).toBeInTheDocument();
  });
});
