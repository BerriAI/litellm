import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";
import { RealtimePrettyView } from "./RealtimePrettyView";

function LanguageSwitch() {
  const { setPreference } = useLanguage();
  return <button onClick={() => setPreference("en")}>English</button>;
}

describe("实时日志缺省文案语言", () => {
  it("仅翻译缺失状态和角色的占位，保留真实值和响应内容", () => {
    const response = {
      results: [
        {
          type: "response.done",
          response: {
            id: "missing-state-role",
            output: [{ id: "missing-role", type: "message", content: [{ type: "text", text: "Home" }] }],
          },
        },
        {
          type: "response.done",
          response: {
            id: "literal-state-role",
            status: "unknown",
            output: [
              { id: "literal-role", role: "assistant", type: "message", content: [{ type: "text", text: "body" }] },
            ],
          },
        },
      ],
    };
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="realtime-fallback" storageScope="realtime-test">
        <LanguageSwitch />
        <RealtimePrettyView response={response} />
      </LanguageProvider>,
    );

    expect(screen.getByText("未知")).toBeInTheDocument();
    expect(screen.getByText("助手")).toBeInTheDocument();
    expect(screen.getByText("unknown")).toBeInTheDocument();
    expect(screen.getByText("ASSISTANT")).toBeInTheDocument();
    expect(screen.getByText("Home")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "English" }));

    expect(screen.getAllByText("unknown")).toHaveLength(2);
    expect(screen.getAllByText("ASSISTANT")).toHaveLength(2);
    expect(screen.getByText("Home")).toBeInTheDocument();
    expect(response.results[0].response.output[0]).not.toHaveProperty("role");
  });
});
