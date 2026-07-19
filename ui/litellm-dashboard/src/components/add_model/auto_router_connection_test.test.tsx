import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import { vi } from "vitest";
import AutoRouterConnectionTest from "./auto_router_connection_test";
import { AutoRouterTestTarget } from "./build_auto_router_test_targets";

vi.mock("../networking", async () => {
  const actual = await vi.importActual("../networking");
  return {
    ...actual,
    testModelGroupConnection: vi.fn(),
  };
});

const getMock = async () => vi.mocked((await import("../networking")).testModelGroupConnection);

const targets: AutoRouterTestTarget[] = [
  { labels: ["SIMPLE"], modelGroup: "gpt-4o-mini", mode: "chat" },
  { labels: ["MEDIUM", "COMPLEX"], modelGroup: "claude-sonnet-4", mode: "chat" },
  { labels: ["Embedding"], modelGroup: "voyage-3-5", mode: "embedding" },
];

describe("AutoRouterConnectionTest", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("probes each target once with the right model and mode (chat for tiers, embedding for the embedding model)", async () => {
    const mock = await getMock();
    mock.mockResolvedValue({ status: "success" });

    renderWithProviders(<AutoRouterConnectionTest accessToken="sk-test" targets={targets} />);

    await waitFor(() => expect(mock).toHaveBeenCalledTimes(3));

    expect(mock).toHaveBeenCalledWith("sk-test", "gpt-4o-mini", "chat");
    expect(mock).toHaveBeenCalledWith("sk-test", "claude-sonnet-4", "chat");
    expect(mock).toHaveBeenCalledWith("sk-test", "voyage-3-5", "embedding");
  });

  it("shows a success indicator per target when the routing probe passes", async () => {
    const mock = await getMock();
    mock.mockResolvedValue({ status: "success" });

    renderWithProviders(<AutoRouterConnectionTest accessToken="sk-test" targets={targets} />);

    await waitFor(() => expect(screen.getAllByTestId("test-status-success")).toHaveLength(3));
    expect(screen.queryByTestId("test-status-error")).toBeNull();
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

    await waitFor(() => expect(screen.getByTestId("test-error-message")).toBeInTheDocument());
    expect(screen.getByTestId("test-error-message")).toHaveTextContent("invalid api key");
    expect(screen.getByTestId("test-error-message")).not.toHaveTextContent("litellm.AuthenticationError");
    expect(screen.getAllByTestId("test-status-success")).toHaveLength(2);
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
