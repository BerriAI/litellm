import { describe, expect, it } from "vitest";
import { englishTranslate, translate } from "@/i18n/language";
import { translateModelLabel, type ModelLabelKind } from "./modelLabels";

const chinese = (message: string) => translate("zh-CN", message);

describe("模型展示标签的翻译边界", () => {
  it.each([
    ["Home", "mode"],
    ["Vision", "mode"],
    ["Settings", "feature"],
    ["chat", "feature"],
  ] as const)("保留 %s 在 %s 字段中的扩展值", (value, kind: ModelLabelKind) => {
    expect(translateModelLabel(value, kind, chinese)).toBe(value);
  });

  it("相同名称只在对应的能力域翻译", () => {
    expect(translateModelLabel("Vision", "feature", chinese)).toBe("视觉");
    expect(translateModelLabel("Vision", "mode", chinese)).toBe("Vision");
  });

  it("已知模式翻译展示标签，但英文保留既有大小写", () => {
    expect(translateModelLabel("chat", "mode", chinese, "Chat")).toBe("对话");
    expect(translateModelLabel("chat", "mode", englishTranslate, "Chat")).toBe("Chat");
    expect(translateModelLabel("chat", "mode", englishTranslate)).toBe("chat");
  });

  it("扩展模式的格式化标签也不会被其他词条误翻译", () => {
    expect(translateModelLabel("home", "mode", chinese, "Home")).toBe("Home");
  });
});
