import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LanguageProvider } from "@/contexts/LanguageContext";
import ContentFilterDetails from "./ContentFilterDetails";

describe("内容检测结果汉化", () => {
  it("汉化已知动作、级别和计数，保留自定义检测内容", () => {
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="content-filter-test" storageScope="content-filter-test">
        <ContentFilterDetails
          response={[
            { type: "pattern", pattern_name: "Home", action: "BLOCK" },
            { type: "category_keyword", category: "Settings", keyword: "Models", severity: "high", action: "MASK" },
          ]}
        />
      </LanguageProvider>,
    );
    expect(screen.getByText("已拦截 1 项")).toBeInTheDocument();
    expect(screen.getByText("已脱敏 1 项")).toBeInTheDocument();
    expect(screen.getByText("拦截")).toBeInTheDocument();
    expect(screen.getByText("脱敏")).toBeInTheDocument();
    expect(screen.getByText("高")).toBeInTheDocument();
    expect(screen.getByText("Home")).toBeInTheDocument();
    expect(screen.getByText("Settings")).toBeInTheDocument();
    expect(screen.getByText("Models")).toBeInTheDocument();
  });

  it("英文动作保留原有大写文本", () => {
    render(<ContentFilterDetails response={[{ type: "pattern", action: "BLOCK" }]} />);
    expect(screen.getByText("BLOCK")).toBeInTheDocument();
    expect(screen.getByText("unknown")).toBeInTheDocument();
  });
});
