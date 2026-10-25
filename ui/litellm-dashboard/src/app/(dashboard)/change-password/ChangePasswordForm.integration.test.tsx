import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ChangePasswordForm from "./ChangePasswordForm";
import { LanguageProvider } from "@/contexts/LanguageContext";
import LanguageSelector from "@/components/LanguageSelector";

const mockChangePasswordCall = vi.fn();
const mockToastSuccess = vi.fn();
const mockClearTokenCookies = vi.fn();
const mockSessionLogoutCall = vi.fn();
let mockPasswordResetRequired = false;

vi.mock("@/components/networking", () => ({
  changePasswordCall: (...args: unknown[]) => mockChangePasswordCall(...args),
  sessionLogoutCall: (...args: unknown[]) => mockSessionLogoutCall(...args),
  getProxyBaseUrl: () => "",
  getUiPageUrl: (url: string) => url,
}));

vi.mock("@/app/(dashboard)/hooks/useAuthorized", () => ({
  default: () => ({ accessToken: "sk-session-token", passwordResetRequired: mockPasswordResetRequired }),
}));

vi.mock("@/lib/toast", () => ({
  toast: {
    success: (...args: unknown[]) => mockToastSuccess(...args),
    fromError: vi.fn(),
  },
}));

vi.mock("@/utils/cookieUtils", () => ({
  clearTokenCookies: (...args: unknown[]) => mockClearTokenCookies(...args),
}));

const fillForm = (values: { current: string; next: string; confirm: string }) => {
  fireEvent.change(screen.getByLabelText("Current Password"), { target: { value: values.current } });
  fireEvent.change(screen.getByLabelText("New Password"), { target: { value: values.next } });
  fireEvent.change(screen.getByLabelText("Confirm New Password"), { target: { value: values.confirm } });
};

const submit = () => fireEvent.click(screen.getByRole("button", { name: "Change Password" }));

describe("ChangePasswordForm", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    mockPasswordResetRequired = false;
    mockSessionLogoutCall.mockResolvedValue({});
  });

  // 已显示的本地校验随语言变化，不能重新提交或清空密码草稿。
  it("切换语言会更新已显示的本地密码校验并保留所有输入", async () => {
    render(
      <LanguageProvider defaultLanguage="en" userId="password-errors" storageScope="management">
        <LanguageSelector />
        <ChangePasswordForm />
      </LanguageProvider>,
    );
    fillForm({ current: "Settings", next: "Home", confirm: "Different" });
    submit();
    expect(await screen.findByText("New passwords do not match")).toBeVisible();

    fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "zh-CN" } });
    expect(await screen.findByText("两次输入的新密码不一致")).toBeVisible();
    expect(screen.getByLabelText("当前密码")).toHaveValue("Settings");
    expect(screen.getByLabelText("新密码")).toHaveValue("Home");
    expect(screen.getByLabelText("确认新密码")).toHaveValue("Different");
    expect(screen.queryByText("请输入当前密码")).not.toBeInTheDocument();
    fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
    expect(await screen.findByText("New passwords do not match")).toBeVisible();
    expect(mockChangePasswordCall).not.toHaveBeenCalled();
  });

  it("切换语言不翻译与本地校验同名的服务器拒绝消息", async () => {
    mockChangePasswordCall.mockRejectedValue(new Error("{'error': 'New passwords do not match'}"));
    render(
      <LanguageProvider defaultLanguage="en" userId="password-server" storageScope="management">
        <LanguageSelector />
        <ChangePasswordForm />
      </LanguageProvider>,
    );
    fillForm({ current: "Settings", next: "Home", confirm: "Home" });
    submit();
    expect(await screen.findByText("New passwords do not match")).toBeVisible();
    fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "zh-CN" } });
    expect(screen.getByText("New passwords do not match")).toBeVisible();
    expect(screen.queryByText("两次输入的新密码不一致")).not.toBeInTheDocument();
    expect(mockChangePasswordCall).toHaveBeenCalledTimes(1);
  });

  it("本地密码校验显示中文，但不翻译服务端拒绝消息", async () => {
    mockChangePasswordCall.mockRejectedValue(new Error("{'error': 'Current password is incorrect.'}"));
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId="password-language" storageScope="management">
        <ChangePasswordForm />
      </LanguageProvider>,
    );
    fireEvent.change(screen.getByLabelText("当前密码"), { target: { value: "Settings" } });
    fireEvent.change(screen.getByLabelText("新密码"), { target: { value: "Home" } });
    fireEvent.change(screen.getByLabelText("确认新密码"), { target: { value: "Different" } });
    fireEvent.click(screen.getByRole("button", { name: "修改密码" }));
    expect(await screen.findByText("两次输入的新密码不一致")).toBeVisible();
    expect(mockChangePasswordCall).not.toHaveBeenCalled();

    fireEvent.change(screen.getByLabelText("确认新密码"), { target: { value: "Home" } });
    fireEvent.click(screen.getByRole("button", { name: "修改密码" }));
    expect(await screen.findByText("Current password is incorrect.")).toBeVisible();
    expect(mockChangePasswordCall).toHaveBeenCalledWith("sk-session-token", "Settings", "Home");
  });

  it("sends the current and new password to the change endpoint and resets on success", async () => {
    mockChangePasswordCall.mockResolvedValue({ user_id: "user-123", message: "Password updated successfully." });
    render(<ChangePasswordForm />);

    fillForm({ current: "OldP@ssw0rd-2026", next: "NewP@ssw0rd-2026", confirm: "NewP@ssw0rd-2026" });
    submit();

    expect(await screen.findByLabelText("Current Password")).toHaveValue("");
    expect(mockChangePasswordCall).toHaveBeenCalledWith("sk-session-token", "OldP@ssw0rd-2026", "NewP@ssw0rd-2026");
    expect(mockToastSuccess).toHaveBeenCalled();
  });

  it("blocks submission when the confirmation does not match", async () => {
    render(<ChangePasswordForm />);

    fillForm({ current: "OldP@ssw0rd-2026", next: "NewP@ssw0rd-2026", confirm: "Different-2026" });
    submit();

    expect(await screen.findByText("New passwords do not match")).toBeInTheDocument();
    expect(mockChangePasswordCall).not.toHaveBeenCalled();
  });

  it("shows the proxy's rejection message unwrapped", async () => {
    mockChangePasswordCall.mockRejectedValue(new Error("{'error': 'Current password is incorrect.'}"));
    render(<ChangePasswordForm />);

    fillForm({ current: "wrong-password", next: "NewP@ssw0rd-2026", confirm: "NewP@ssw0rd-2026" });
    submit();

    expect(await screen.findByText("Current password is incorrect.")).toBeInTheDocument();
    expect(mockToastSuccess).not.toHaveBeenCalled();
  });

  describe("forced password reset", () => {
    it("shows the forced-reset warning only when the session is flagged", () => {
      mockPasswordResetRequired = true;
      render(<ChangePasswordForm />);

      expect(screen.getByText(/must be changed before you can use the dashboard/)).toBeInTheDocument();
    });

    it("hides the forced-reset warning for a normal session", () => {
      render(<ChangePasswordForm />);

      expect(screen.queryByText(/must be changed before you can use the dashboard/)).not.toBeInTheDocument();
    });

    it("signs the user out to re-login after a successful forced change", async () => {
      mockPasswordResetRequired = true;
      mockChangePasswordCall.mockResolvedValue({ user_id: "user-123", message: "Password updated successfully." });
      const replaceMock = vi.fn();
      const realLocation = window.location;
      Object.defineProperty(window, "location", { configurable: true, value: { replace: replaceMock } });

      try {
        render(<ChangePasswordForm />);
        fillForm({ current: "OldP@ssw0rd-2026", next: "NewP@ssw0rd-2026", confirm: "NewP@ssw0rd-2026" });
        submit();

        await waitFor(() => expect(replaceMock).toHaveBeenCalledWith("/ui/login/"));
        expect(mockSessionLogoutCall).toHaveBeenCalledWith("sk-session-token");
        expect(mockClearTokenCookies).toHaveBeenCalled();
        // 翻译通知不得改变强制改密后的服务端注销、清理与跳转顺序。
        expect(mockSessionLogoutCall.mock.invocationCallOrder[0]).toBeLessThan(
          mockClearTokenCookies.mock.invocationCallOrder[0],
        );
        expect(mockClearTokenCookies.mock.invocationCallOrder[0]).toBeLessThan(replaceMock.mock.invocationCallOrder[0]);
      } finally {
        Object.defineProperty(window, "location", { configurable: true, value: realLocation });
      }
    });
  });
});
