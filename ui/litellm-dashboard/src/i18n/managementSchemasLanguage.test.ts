import { describe, expect, it } from "vitest";
import {
  buildProjectFormSchema,
  emptyProjectFormValues,
} from "@/app/(dashboard)/projects/_components/ProjectModals/projectFormSchema";
import { buildDefaultUserSettingsSchema } from "@/app/(dashboard)/users/_components/default-user-settings/schema";
import { buildPluginSchema } from "@/components/Settings/AdminSettings/PluginSettings/schema";
import { buildAccessGroupCreateSchema } from "@/app/(dashboard)/access-groups/_components/access-group-create/schema";
import { translate, type Translate } from "./language";

const chinese: Translate = (message, values) => translate("zh-CN", message, values);

describe("管理表单校验语言", () => {
  it("默认用户的本地失败提示确实提供中文，不以英文回退充当翻译", () => {
    // 使用独立中文预期，避免词典缺项时测试和界面同时回退英文而误通过。
    expect(chinese("Failed to update default user settings")).toBe("更新默认用户设置失败");
  });

  it("访问组名称校验使用中文，资源 ID 和说明不翻译", () => {
    const schema = buildAccessGroupCreateSchema(chinese);
    const values = {
      name: "Home",
      description: "Settings",
      modelIds: ["Home"],
      mcpServerIds: ["Settings"],
      mcpToolPermissions: {},
      agentIds: [],
      skillNames: [],
    };
    expect(schema.parse(values)).toStrictEqual(values);
    const result = schema.safeParse({ ...values, name: " " });
    expect(result.success).toBe(false);
    if (!result.success) expect(result.error.issues[0].message).toBe("请输入访问组名称");
  });
  it("项目校验翻译重复模型提示，但不改模型名或团队 ID", () => {
    const schema = buildProjectFormSchema(chinese);
    const duplicateValues = {
      ...emptyProjectFormValues,
      project_alias: "Home",
      team_id: "Settings",
      modelLimits: [{ model: "Home" }, { model: "Home" }],
    };
    const duplicate = schema.safeParse(duplicateValues);
    expect(duplicate.success).toBe(false);
    if (!duplicate.success) {
      expect(duplicate.error.issues).toEqual([
        expect.objectContaining({ path: ["modelLimits", 1, "model"], message: chinese("Duplicate model") }),
      ]);
    }
    const validValues = { ...emptyProjectFormValues, project_alias: "Home", team_id: "Settings", models: ["Home"] };
    expect(schema.parse(validValues)).toStrictEqual(validValues);
  });

  it("默认用户配置只翻译校验提示，角色值保持原样", () => {
    const schema = buildDefaultUserSettingsSchema(chinese);
    const values = { user_role: "internal_user", max_budget: "", budget_duration: "30d", models: ["Home"], teams: [] };
    expect(schema.parse(values)).toStrictEqual(values);
    const result = schema.safeParse({
      ...values,
      teams: [
        { team_id: "Home", max_budget_in_team: "", user_role: "user" },
        { team_id: "Home", max_budget_in_team: "", user_role: "admin" },
      ],
    });
    expect(result.success).toBe(false);
    if (!result.success) expect(result.error.issues[0].message).toBe(chinese("This team is already listed"));
  });

  it("插件 URL 的中文校验保留默认英文 schema 兼容性", () => {
    const values = { name: "Home", display_name: "Settings", url: "invalid" };
    const localized = buildPluginSchema(chinese).safeParse(values);
    const english = buildPluginSchema().safeParse(values);
    expect(localized.success).toBe(false);
    expect(english.success).toBe(false);
    if (!localized.success) expect(localized.error.issues[0].message).toBe("请输入有效的 URL");
    if (!english.success) expect(english.error.issues[0].message).toBe("Must be a valid URL");
  });
});
