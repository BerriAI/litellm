import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { OnboardingFormBody } from "./OnboardingFormBody";
import { LanguageProvider } from "@/contexts/LanguageContext";

const defaultProps = {
  variant: "signup" as const,
  userEmail: "test@example.com",
  isPending: false,
  claimError: null,
};

describe("OnboardingFormBody submit payload", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  it.each(["signup", "reset_password"] as const)(
    "%s 的已显示字段校验随语言更新，服务端同名错误保持原文",
    async (variant) => {
      const onSubmit = vi.fn();
      render(
        <LanguageProvider defaultLanguage="en" userId={null} storageScope="invite-validation">
          <OnboardingFormBody
            {...defaultProps}
            variant={variant}
            claimError="password required to sign up"
            onSubmit={onSubmit}
          />
        </LanguageProvider>,
      );
      fireEvent.click(screen.getByRole("button", { name: variant === "signup" ? "Sign Up" : "Reset Password" }));
      await waitFor(() => expect(screen.getByLabelText("Password")).toHaveAttribute("aria-invalid", "true"));
      fireEvent.change(screen.getByRole("combobox", { name: "Interface language" }), { target: { value: "zh-CN" } });
      expect(await screen.findByText("请输入密码以完成注册")).toBeVisible();
      expect(screen.getByText("password required to sign up")).toBeVisible();
      expect(screen.getByLabelText("邮箱地址")).toHaveValue("test@example.com");
      fireEvent.change(screen.getByRole("combobox", { name: "界面语言" }), { target: { value: "en" } });
      expect(await screen.findAllByText("password required to sign up")).toHaveLength(2);
      expect(onSubmit).not.toHaveBeenCalled();
    },
  );

  it("sends exactly {password} and never leaks the email field", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<OnboardingFormBody {...defaultProps} onSubmit={onSubmit} />);

    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "hunter2" } });
    await user.click(screen.getByRole("button", { name: "Sign Up" }));

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0]).toStrictEqual({ password: "hunter2" });
  });

  it("中文邀请表单保留邮箱、后端错误及密码 payload", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <LanguageProvider defaultLanguage="zh-CN" userId={null} storageScope="invite-test">
        <OnboardingFormBody {...defaultProps} userEmail="Home@example.com" claimError="Settings" onSubmit={onSubmit} />
      </LanguageProvider>,
    );
    expect(screen.getByLabelText("邮箱地址")).toHaveValue("Home@example.com");
    expect(screen.getByText("Settings")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "注册" }));
    expect(await screen.findByText("请输入密码以完成注册")).toBeVisible();
    expect(onSubmit).not.toHaveBeenCalled();
    fireEvent.change(screen.getByLabelText("密码", { exact: true }), { target: { value: "Settings" } });
    await user.click(screen.getByRole("button", { name: "注册" }));
    await waitFor(() => expect(onSubmit).toHaveBeenCalledWith({ password: "Settings" }));
    await user.selectOptions(screen.getByRole("combobox", { name: "界面语言" }), "en");
    expect(screen.getByRole("heading", { name: "Sign Up" })).toBeVisible();
    expect(screen.getByLabelText("Password", { exact: true })).toHaveValue("Settings");
  });

  it("submits on Enter from the password field", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<OnboardingFormBody {...defaultProps} onSubmit={onSubmit} />);

    await user.type(screen.getByLabelText("Password"), "hunter2{Enter}");

    await waitFor(() => expect(onSubmit).toHaveBeenCalledTimes(1));
    expect(onSubmit.mock.calls[0][0]).toStrictEqual({ password: "hunter2" });
  });

  it("blocks submit and marks the password invalid when it is empty", async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<OnboardingFormBody {...defaultProps} onSubmit={onSubmit} />);

    await user.click(screen.getByRole("button", { name: "Sign Up" }));

    await waitFor(() => expect(screen.getByLabelText("Password")).toHaveAttribute("aria-invalid", "true"));
    expect(screen.getByText("Create a password for your account")).toBeInTheDocument();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("keeps the password help text always visible and variant specific", async () => {
    const { unmount } = render(<OnboardingFormBody {...defaultProps} onSubmit={vi.fn()} />);
    expect(screen.getByText("Create a password for your account")).toBeInTheDocument();
    unmount();

    render(<OnboardingFormBody {...defaultProps} variant="reset_password" onSubmit={vi.fn()} />);
    expect(screen.getByText("Enter your new password")).toBeInTheDocument();
  });
});
