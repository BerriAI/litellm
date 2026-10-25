import { act, fireEvent, renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import { vi } from "vitest";
import AutoRouterConnectionTest from "./auto_router_connection_test";
import { AutoRouterTestTarget } from "./build_auto_router_test_targets";
import { LanguageProvider, useLanguage } from "@/contexts/LanguageContext";
import { testAutoRouterRouting, type AutoRouterRoutingTestResponse } from "../networking";

vi.mock("../networking", async () => {
  const actual = await vi.importActual("../networking");
  return {
    ...actual,
    testModelGroupConnection: vi.fn(),
    testAutoRouterRouting: vi.fn(),
  };
});

const getMock = async () => vi.mocked((await import("../networking")).testModelGroupConnection);

const targets: AutoRouterTestTarget[] = [
  { labels: ["SIMPLE"], modelGroup: "gpt-4o-mini", mode: "chat" },
  { labels: ["MEDIUM", "COMPLEX"], modelGroup: "claude-sonnet-4", mode: "chat" },
  { labels: ["Embedding"], modelGroup: "voyage-3-5", mode: "embedding" },
  { labels: ["Classifier"], modelGroup: "gpt-5-mini", mode: "chat", requestParams: { reasoning_effort: "low" } },
];

describe("AutoRouterConnectionTest", () => {
  it("OSS 本地测试状态随语言切换，保留路由原因且不重复发送请求", async () => {
    let complete: (result: AutoRouterRoutingTestResponse) => void = () => undefined;
    vi.mocked(testAutoRouterRouting).mockReturnValue(
      new Promise((resolve) => {
        complete = resolve;
      }),
    );
    function SwitchLanguage() {
      const { setPreference } = useLanguage();
      return <button onClick={() => setPreference("en")}>English</button>;
    }
    renderWithProviders(
      <LanguageProvider defaultLanguage="zh-CN" userId="oss-test" storageScope="oss-test-language">
        <SwitchLanguage />
        <AutoRouterConnectionTest
          accessToken="token"
          targets={[]}
          jevRequest={{ prompt: "Home", complexity_router_config: {} }}
        />
      </LanguageProvider>,
    );
    expect(screen.getByRole("status")).toHaveTextContent("正在测试 OSS 分类");
    await act(async () =>
      complete({
        status: "success",
        result: {
          routed_model: "Home",
          routed_model_configured: true,
          routing_decision: {
            router_model_name: "Home",
            router_type: "complexity",
            routed_model: "Home",
            cause: "Home",
          },
        },
      }),
    );
    expect(screen.getByRole("status")).toHaveTextContent("未能成功调用 OSS 分类器（路由原因：Home）");
    fireEvent.click(screen.getByRole("button", { name: "English" }));
    expect(screen.getByRole("status")).toHaveTextContent(
      "OSS classifier was not reached successfully (routing cause: Home)",
    );
    expect(testAutoRouterRouting).toHaveBeenCalledOnce();
  });

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("probes each target once with the right model and mode (chat for tiers, embedding for the embedding model)", async () => {
    const mock = await getMock();
    mock.mockResolvedValue({ status: "success" });

    renderWithProviders(<AutoRouterConnectionTest accessToken="sk-test" targets={targets} />);

    await waitFor(() => expect(mock).toHaveBeenCalledTimes(4));

    expect(mock).toHaveBeenCalledWith("sk-test", "gpt-4o-mini", "chat");
    expect(mock).toHaveBeenCalledWith("sk-test", "claude-sonnet-4", "chat");
    expect(mock).toHaveBeenCalledWith("sk-test", "voyage-3-5", "embedding");
    expect(mock).toHaveBeenCalledWith("sk-test", "gpt-5-mini", "chat", { reasoning_effort: "low" });
  });

  it("shows a success indicator per target when the routing probe passes", async () => {
    const mock = await getMock();
    mock.mockResolvedValue({ status: "success" });

    renderWithProviders(<AutoRouterConnectionTest accessToken="sk-test" targets={targets} />);

    await waitFor(() => expect(screen.getAllByTestId("test-status-success")).toHaveLength(4));
    expect(screen.queryByTestId("test-status-error")).not.toBeInTheDocument();
    expect(screen.getByText("MEDIUM, COMPLEX")).toBeInTheDocument();
  });

  it("renders the provider error message (litellm prefix stripped) for a failing target while others pass", async () => {
    const mock = await getMock();
    mock.mockImplementation((_token, modelGroup) =>
      Promise.resolve(
        modelGroup === "claude-sonnet-4"
          ? { status: "error", error: "litellm.AuthenticationError: invalid api key" }
          : { status: "success" },
      ),
    );

    renderWithProviders(<AutoRouterConnectionTest accessToken="sk-test" targets={targets} />);

    expect(await screen.findByTestId("test-error-message")).toBeInTheDocument();
    expect(screen.getByTestId("test-error-message")).toHaveTextContent("invalid api key");
    expect(screen.getByTestId("test-error-message")).not.toHaveTextContent("litellm.AuthenticationError");
    expect(screen.getAllByTestId("test-status-success")).toHaveLength(3);
  });

  it("renders a non-litellm error string verbatim", async () => {
    const mock = await getMock();
    mock.mockResolvedValue({ status: "error", error: "Connection test failed: 404 Not Found" });

    renderWithProviders(
      <AutoRouterConnectionTest
        accessToken="sk-test"
        targets={[{ labels: ["SIMPLE"], modelGroup: "gpt-4o-mini", mode: "chat" }]}
      />,
    );

    await waitFor(() =>
      expect(screen.getByTestId("test-error-message")).toHaveTextContent("Connection test failed: 404 Not Found"),
    );
  });
});
