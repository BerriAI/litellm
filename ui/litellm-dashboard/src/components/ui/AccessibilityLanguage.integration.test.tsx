import { createRef } from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import LanguageSelector from "@/components/LanguageSelector";
import { LanguageProvider } from "@/contexts/LanguageContext";
import { Breadcrumb, BreadcrumbItem, BreadcrumbList, BreadcrumbPage } from "./breadcrumb";
import { SimpleTooltip } from "./tooltip";

const languageTree = (children: React.ReactNode) => (
  <LanguageProvider defaultLanguage="zh-CN" userId="accessibility-review" storageScope="/accessibility-review">
    <LanguageSelector />
    {children}
  </LanguageProvider>
);

// 中文预期不读取词典，避免组件与测试同时回退英文而误通过。
describe("公共控件的可访问名称", () => {
  beforeEach(() => localStorage.clear());

  it("导航名称随语言切换，当前页和原始业务内容不变", () => {
    const navigationRef = createRef<HTMLElement>();
    render(
      languageTree(
        <Breadcrumb ref={navigationRef}>
          <BreadcrumbList>
            <BreadcrumbItem>
              <BreadcrumbPage>Home</BreadcrumbPage>
            </BreadcrumbItem>
          </BreadcrumbList>
        </Breadcrumb>,
      ),
    );

    const navigation = screen.getByRole("navigation", { name: "页面导航" });
    expect(navigationRef.current).toBe(navigation);
    expect(screen.getByRole("link", { name: "Home" })).toHaveAttribute("aria-current", "page");

    fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });

    expect(screen.getByRole("navigation", { name: "Breadcrumb" })).toBe(navigation);
    expect(navigationRef.current).toBe(navigation);
    expect(screen.getByRole("link", { name: "Home" })).toHaveAttribute("aria-current", "page");
  });

  it("保留调用方指定的导航标签", () => {
    render(languageTree(<Breadcrumb aria-label="Home" />));
    expect(screen.getByRole("navigation", { name: "Home" })).toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "en" } });
    expect(screen.getByRole("navigation", { name: "Home" })).toBeInTheDocument();
  });

  it("帮助图标随语言切换，已打开的提示保留调用方原文", async () => {
    const user = userEvent.setup();
    render(languageTree(<SimpleTooltip content="Home" />));
    const icon = screen.getByLabelText("帮助");
    await user.hover(icon);
    expect(await screen.findByText("Home")).toBeInTheDocument();

    fireEvent.change(screen.getByRole("combobox"), { target: { value: "en" } });

    expect(screen.getByLabelText("Help")).toBe(icon);
    expect(screen.getByText("Home")).toBeInTheDocument();
    await user.unhover(icon);
    expect(screen.queryByText("Home")).not.toBeInTheDocument();
  });

  it.each([undefined, null, ""])("空提示 %s 的默认图标仍使用当前语言", (content) => {
    render(languageTree(<SimpleTooltip content={content} />));
    expect(screen.getByLabelText("帮助")).toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "en" } });
    expect(screen.getByLabelText("Help")).toBeInTheDocument();
  });

  it("自定义触发器的业务名称不进入词典", () => {
    render(
      languageTree(
        <SimpleTooltip content="Home">
          <button type="button" aria-label="Home">
            Home
          </button>
        </SimpleTooltip>,
      ),
    );
    expect(screen.getByRole("button", { name: "Home" })).toBeInTheDocument();
    expect(screen.queryByLabelText("帮助")).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox"), { target: { value: "en" } });
    expect(screen.getByRole("button", { name: "Home" })).toBeInTheDocument();
    expect(screen.queryByLabelText("Help")).not.toBeInTheDocument();
  });
});
