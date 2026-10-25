import { describe, expect, it } from "vitest";
import { resolveLanguage, translate } from "./language";

describe("语言回退与翻译", () => {
  it("汉化上游新增的批量编辑与模型空状态，英文协议文案不变", () => {
    expect(translate("zh-CN", "No change")).toBe("不修改");
    expect(translate("en", "No change")).toBe("No change");
    expect(translate("zh-CN", "No available models")).toBe("没有可用模型");
    expect(translate("en", "No available models")).toBe("No available models");
  });

  it("个人选择优先于全站默认", () => {
    expect(resolveLanguage("en", "zh-CN")).toBe("en");
    expect(resolveLanguage("zh-CN", "en")).toBe("zh-CN");
  });

  it("无个人选择时使用全站默认，非法配置回退英文", () => {
    expect(resolveLanguage(null, "zh-CN")).toBe("zh-CN");
    expect(resolveLanguage("unknown", "zh-CN")).toBe("zh-CN");
    expect(resolveLanguage(null, "unknown")).toBe("en");
  });

  it("翻译静态文案，保留缺失翻译，并安全插入参数", () => {
    expect(translate("zh-CN", "Home")).toBe("首页");
    expect(translate("en", "Home")).toBe("Home");
    expect(translate("zh-CN", "untranslated message")).toBe("untranslated message");
    expect(translate("zh-CN", "__proto__")).toBe("__proto__");
    expect(translate("zh-CN", "constructor")).toBe("constructor");
    expect(translate("zh-CN", "{constructor} {missing}")).toBe("{constructor} {missing}");
    expect(translate("zh-CN", "Usage, last {days} days", { days: 7 })).toBe("最近 7 天的用量");
    expect(translate("zh-CN", "Account menu, {identity}", { identity: "$& <admin>" })).toBe("账户菜单，$& <admin>");
  });
});
