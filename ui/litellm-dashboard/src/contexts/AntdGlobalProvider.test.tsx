import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { I18nProvider } from "@/i18n/I18nProvider";
import { LANGUAGE_STORAGE_KEY } from "@/i18n/language";
// eslint-disable-next-line no-restricted-imports -- verifies that the compatibility provider receives the real Russian antd locale
import ruRU from "antd/locale/ru_RU";
import AntdGlobalProvider from "./AntdGlobalProvider";

const { configProviderMock } = vi.hoisted(() => ({ configProviderMock: vi.fn() }));

vi.mock("antd", () => ({
  ConfigProvider: ({ children, ...props }: { children: React.ReactNode; locale?: unknown }) => {
    configProviderMock(props);
    return <div data-testid="config-provider">{children}</div>;
  },
  notification: { useNotification: () => [{}, null] },
  message: { useMessage: () => [{}, null] },
}));

vi.mock("@ant-design/cssinjs", () => ({
  StyleProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock("@/components/molecules/notifications_manager", () => ({ setNotificationInstance: vi.fn() }));
vi.mock("@/components/molecules/message_manager", () => ({ setMessageInstance: vi.fn() }));

describe("AntdGlobalProvider", () => {
  beforeEach(() => {
    configProviderMock.mockClear();
    window.localStorage.setItem(LANGUAGE_STORAGE_KEY, "ru");
  });

  it("passes the active Russian locale to Ant Design", async () => {
    render(
      <I18nProvider>
        <AntdGlobalProvider>
          <div>content</div>
        </AntdGlobalProvider>
      </I18nProvider>,
    );

    expect(await screen.findByText("content")).toBeInTheDocument();
    expect(configProviderMock).toHaveBeenLastCalledWith(expect.objectContaining({ locale: ruRU }));
  });
});
