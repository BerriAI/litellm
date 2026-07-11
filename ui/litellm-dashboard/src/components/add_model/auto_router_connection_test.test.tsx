import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import { vi } from "vitest";
import AutoRouterConnectionTest from "./auto_router_connection_test";
import { AutoRouterTestTarget } from "./build_auto_router_test_targets";

vi.mock("../networking", async () => {
  const actual = await vi.importActual("../networking");
  return {
    ...actual,
    testConnectionRequest: vi.fn(),
  };
});

const getMock = async () => vi.mocked((await import("../networking")).testConnectionRequest);

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

    expect(mock).toHaveBeenCalledWith("sk-test", { model: "gpt-4o-mini" }, {}, "chat");
    expect(mock).toHaveBeenCalledWith("sk-test", { model: "claude-sonnet-4" }, {}, "chat");
    expect(mock).toHaveBeenCalledWith("sk-test", { model: "voyage-3-5" }, {}, "embedding");
  });

  it("shows a success indicator per target when the health check passes", async () => {
    const mock = await getMock();
    mock.mockResolvedValue({ status: "success" });

    renderWithProviders(<AutoRouterConnectionTest accessToken="sk-test" targets={targets} />);

    await waitFor(() => expect(screen.getAllByTestId("test-status-success")).toHaveLength(3));
    expect(screen.queryByTestId("test-status-error")).toBeNull();
    expect(screen.getByText("MEDIUM, COMPLEX")).toBeInTheDocument();
  });

  it("renders the provider error message for a failing target while others pass", async () => {
    const mock = await getMock();
    mock.mockImplementation((_token, litellmParams) =>
      litellmParams.model === "claude-sonnet-4"
        ? Promise.resolve({ status: "error", result: { error: "litellm.AuthenticationError: invalid api key" } })
        : Promise.resolve({ status: "success" }),
    );

    renderWithProviders(<AutoRouterConnectionTest accessToken="sk-test" targets={targets} />);

    await waitFor(() => expect(screen.getByTestId("test-error-message")).toBeInTheDocument());
    expect(screen.getByTestId("test-error-message")).toHaveTextContent("invalid api key");
    expect(screen.getByTestId("test-error-message")).not.toHaveTextContent("litellm.AuthenticationError");
    expect(screen.getAllByTestId("test-status-success")).toHaveLength(2);
  });

  it("surfaces a thrown network error as a failing row", async () => {
    const mock = await getMock();
    mock.mockRejectedValue(new Error("Network request failed"));

    renderWithProviders(
      <AutoRouterConnectionTest
        accessToken="sk-test"
        targets={[{ labels: ["SIMPLE"], modelGroup: "gpt-4o-mini", mode: "chat" }]}
      />,
    );

    await waitFor(() => expect(screen.getByTestId("test-error-message")).toHaveTextContent("Network request failed"));
  });
});
