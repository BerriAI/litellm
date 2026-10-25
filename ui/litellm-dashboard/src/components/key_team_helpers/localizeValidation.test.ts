import { describe, expect, it } from "vitest";
import { translate } from "@/i18n/language";
import { localizeValidation } from "./localizeValidation";

describe("本地模型表单校验翻译", () => {
  it("翻译同步和异步校验提示，不改变有效值或未知提示", async () => {
    const rules = localizeValidation<string, { model: string }>(
      {
        required: (value) => value !== "" || "Required",
        positive: async (value) => Number(value) > 0 || "Enter a positive integer",
        custom: () => "unrecognized-local-message",
      },
      (message, values) => translate("zh-CN", message, values),
    );
    await expect(rules.required("", { model: "model-original-id" })).resolves.toBe("必填");
    await expect(rules.positive("0", { model: "model-original-id" })).resolves.toBe("请输入正整数");
    await expect(rules.required("1", { model: "model-original-id" })).resolves.toBe(true);
    await expect(rules.custom("1", { model: "model-original-id" })).resolves.toBe("unrecognized-local-message");
  });

  it("英文回退保留原校验文案", async () => {
    const rules = localizeValidation<string, { model: string }>({ required: () => "Required" }, (message, values) =>
      translate("en", message, values),
    );
    await expect(rules.required("", { model: "model-original-id" })).resolves.toBe("Required");
  });
});
