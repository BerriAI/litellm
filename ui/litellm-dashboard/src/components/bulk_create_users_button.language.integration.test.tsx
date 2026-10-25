import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";
import { toast } from "@/lib/toast";
import BulkCreateUsersButton from "./bulk_create_users_button";
import { getProxyUISettings, userCreateCall } from "./networking";

vi.mock("./networking", () => ({
  userCreateCall: vi.fn(),
  invitationCreateCall: vi.fn(),
  getProxyUISettings: vi.fn().mockResolvedValue({ SSO_ENABLED: true }),
  getUiPageUrl: (url: string) => url,
}));

function LanguageSwitch() {
  const { language, setPreference } = useLanguage();
  return <button onClick={() => setPreference(language === "en" ? "zh-CN" : "en")}>切换测试语言</button>;
}

function openUpload(defaultLanguage = "zh-CN") {
  render(
    <LanguageProvider defaultLanguage={defaultLanguage} userId="bulk-users-language" storageScope="test">
      <BulkCreateUsersButton accessToken="test-token" teams={[{ team_id: "team-1" }]} possibleUIRoles={null} />
      <LanguageSwitch />
    </LanguageProvider>,
  );
  fireEvent.click(screen.getByRole("button", { name: /Bulk Invite Users|批量邀请用户/ }));
}

function upload(contents: string, name = "users.csv", type = "text/csv") {
  fireEvent.change(screen.getByLabelText(/Drag and drop your CSV file here|CSV.*拖|拖.*CSV/), {
    target: { files: [new File([contents], name, { type })] },
  });
}

// 本地校验按当前语言展示，CSV 字段、用户数据和后端错误不参与翻译。
describe("批量用户导入语言", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.clearAllMocks();
  });

  it("切换语言时更新已有的文件错误，不重复加载或创建用户", async () => {
    openUpload("en");
    upload("invalid", "Settings.txt", "text/plain");
    expect(await screen.findByText(/Invalid file type: Settings.txt/)).toBeVisible();
    fireEvent.click(screen.getByText("切换测试语言"));
    expect(screen.getByText("文件类型不正确：Settings.txt。请上传 CSV 文件（.csv 扩展名）。")).toBeVisible();
    expect(getProxyUISettings).toHaveBeenCalledTimes(1);
    expect(userCreateCall).not.toHaveBeenCalled();
  });

  it("翻译缺失列提示，保留 CSV 列名", async () => {
    openUpload();
    upload("email,role\nuser@example.com,internal_user");
    expect(await screen.findByText("CSV 缺少以下必填列：user_email, user_role。请在文件中添加这些列。")).toBeVisible();
    expect(screen.getByText("teams")).toBeVisible();
    expect(screen.getByText("models")).toBeVisible();
  });

  it("翻译本地行校验且切换语言不丢失已解析数据", async () => {
    openUpload();
    upload("user_email,user_role,max_budget,budget_duration,teams\ninvalid,unknown,NaN,wrong,Settings");
    expect(await screen.findByText("CSV 中没有有效用户。请根据下方错误修改文件。")).toBeVisible();
    const rowError = screen.getByText(/邮箱格式不正确/);
    expect(rowError).toHaveTextContent("角色“unknown”无效");
    expect(rowError).toHaveTextContent("预算上限“NaN”必须为数字");
    expect(rowError).toHaveTextContent("预算周期“wrong”格式不正确");
    expect(rowError).toHaveTextContent("未知团队：Settings");
    fireEvent.click(screen.getByText("切换测试语言"));
    expect(screen.getByText(/Invalid email format/)).toBeVisible();
    expect(screen.getByText("invalid")).toBeVisible();
    expect(getProxyUISettings).toHaveBeenCalledTimes(1);
  });

  it("翻译有效用户数量，发送原始角色、模型和团队值", async () => {
    vi.mocked(userCreateCall).mockResolvedValue({ user_id: "created-user" });
    openUpload();
    upload("user_email,user_role,teams,models\nuser@example.com,internal_user,team-1,Settings");
    expect(await screen.findByText("user@example.com")).toBeVisible();
    expect(toast.success).toHaveBeenCalledWith("已成功解析 1 位用户");
    fireEvent.click(screen.getAllByRole("button", { name: "创建 1 位用户" })[0]);
    const expectedUser = {
      user_email: "user@example.com",
      user_role: "internal_user",
      teams: ["team-1"],
      models: ["Settings"],
    };
    await waitFor(() => expect(userCreateCall).toHaveBeenCalledWith("test-token", null, expectedUser));
  });

  it("后端错误即使与词条相同也保留原文", async () => {
    vi.mocked(userCreateCall).mockResolvedValue({ error: "Settings" });
    openUpload();
    upload("user_email,user_role\nuser@example.com,internal_user");
    fireEvent.click((await screen.findAllByRole("button", { name: "创建 1 位用户" }))[0]);
    expect(await screen.findByText('"Settings"')).toBeVisible();
  });
});
