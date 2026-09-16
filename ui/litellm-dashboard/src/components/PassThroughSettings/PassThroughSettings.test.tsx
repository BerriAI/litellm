import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { act, renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import { deletePassThroughEndpointsCall, getPassThroughEndpointsCall } from "../networking";
import PassThroughSettings from "./PassThroughSettings";
import type { PassThroughEndpointsTable } from "./PassThroughEndpointsTable";

vi.mock("../networking", () => ({
  getPassThroughEndpointsCall: vi.fn(),
  deletePassThroughEndpointsCall: vi.fn(),
}));

vi.mock("../add_pass_through", () => ({
  default: () => <div data-testid="add-pass-through" />,
}));

vi.mock("../pass_through_info", () => ({
  default: ({
    endpointData,
    onClose,
    isAdmin,
  }: {
    endpointData: { id?: string; path: string };
    onClose: () => void;
    isAdmin: boolean;
  }) => (
    <div data-testid="endpoint-info" data-path={endpointData.path} data-admin={String(isAdmin)}>
      {endpointData.id}
      <button type="button" onClick={onClose}>
        close-endpoint
      </button>
    </div>
  ),
}));

vi.mock("./PassThroughEndpointsTable", () => ({
  PassThroughEndpointsTable: (props: React.ComponentProps<typeof PassThroughEndpointsTable>) => (
    <div data-testid="endpoints-table" data-loading={props.isLoading}>
      {props.endpoints.map((endpoint) => (
        <div key={endpoint.id}>
          <button type="button" onClick={() => endpoint.id && props.onEndpointClick(endpoint.id)}>
            open-{endpoint.id}
          </button>
          <button type="button" onClick={() => endpoint.id && props.onDeleteClick(endpoint.id)}>
            delete-{endpoint.id}
          </button>
        </div>
      ))}
    </div>
  ),
}));

const mockGetEndpoints = vi.mocked(getPassThroughEndpointsCall);
const mockDeleteEndpoint = vi.mocked(deletePassThroughEndpointsCall);

const defaultProps = {
  accessToken: "token",
  userRole: "Admin",
  userID: "user-1",
  premiumUser: false,
};

const endpoint = { id: "ep-1", path: "/v1/rerank", target: "https://example.com", headers: {} };

describe("PassThroughSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetEndpoints.mockResolvedValue({ endpoints: [endpoint] });
  });

  it("should render nothing without an access token", () => {
    const { container } = renderWithProviders(<PassThroughSettings {...defaultProps} accessToken={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("should hold the table in loading state until the fetch settles", async () => {
    let resolveEndpoints: (value: { endpoints: (typeof endpoint)[] }) => void = () => {};
    mockGetEndpoints.mockReturnValue(
      new Promise((resolve) => {
        resolveEndpoints = resolve;
      }),
    );

    renderWithProviders(<PassThroughSettings {...defaultProps} />);
    expect(screen.getByTestId("endpoints-table")).toHaveAttribute("data-loading", "true");

    await act(async () => {
      resolveEndpoints({ endpoints: [endpoint] });
    });

    await waitFor(() => {
      expect(screen.getByTestId("endpoints-table")).toHaveAttribute("data-loading", "false");
    });
  });

  it("should resolve loading without fetching when the user id is missing", async () => {
    renderWithProviders(<PassThroughSettings {...defaultProps} userID={null} />);

    await waitFor(() => {
      expect(screen.getByTestId("endpoints-table")).toHaveAttribute("data-loading", "false");
    });
    expect(mockGetEndpoints).not.toHaveBeenCalled();
  });

  it("should swap to the endpoint info view when an endpoint is opened", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PassThroughSettings {...defaultProps} />);

    await user.click(await screen.findByText("open-ep-1"));
    expect(screen.getByTestId("endpoint-info")).toHaveTextContent("ep-1");
  });

  it("should confirm before deleting an endpoint", async () => {
    const user = userEvent.setup();
    mockDeleteEndpoint.mockResolvedValue(undefined);
    renderWithProviders(<PassThroughSettings {...defaultProps} />);

    await user.click(await screen.findByText("delete-ep-1"));
    expect(screen.getByText("Delete Pass-Through Endpoint")).toBeInTheDocument();
    expect(mockDeleteEndpoint).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Delete" }));
    await waitFor(() => {
      expect(mockDeleteEndpoint).toHaveBeenCalledWith("token", "ep-1");
    });
  });

  describe("endpoint in the URL", () => {
    const lastUrlUpdate = (onUrlUpdate: ReturnType<typeof vi.fn<OnUrlUpdateFunction>>) => {
      const lastCall = onUrlUpdate.mock.calls.at(-1);
      if (!lastCall) throw new Error("expected a URL update");
      return lastCall[0];
    };

    it("opens the endpoint named in the URL once loaded, without flashing not found first", async () => {
      let resolveEndpoints: (value: { endpoints: (typeof endpoint)[] }) => void = () => {};
      mockGetEndpoints.mockReturnValue(
        new Promise((resolve) => {
          resolveEndpoints = resolve;
        }),
      );

      renderWithProviders(<PassThroughSettings {...defaultProps} />, { searchParams: "?endpoint=ep-1" });

      expect(screen.getByText("Loading...")).toBeInTheDocument();
      expect(screen.queryByText("Endpoint not found")).not.toBeInTheDocument();
      expect(screen.queryByTestId("endpoints-table")).not.toBeInTheDocument();

      await act(async () => {
        resolveEndpoints({ endpoints: [endpoint] });
      });

      expect(await screen.findByTestId("endpoint-info")).toHaveTextContent("ep-1");
    });

    it("resolves a config endpoint without an id by its path", async () => {
      const configEndpoint = { path: "/from-config", target: "https://config.example.com", headers: {} };
      mockGetEndpoints.mockResolvedValue({ endpoints: [endpoint, configEndpoint] });

      renderWithProviders(<PassThroughSettings {...defaultProps} />, { searchParams: "?endpoint=%2Ffrom-config" });

      expect(await screen.findByTestId("endpoint-info")).toHaveAttribute("data-path", "/from-config");
    });

    it("says the endpoint is not found when the URL names one that does not exist, with a way back", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<PassThroughSettings {...defaultProps} />, {
        searchParams: "?endpoint=ep-gone&endpoint_tab=settings",
        onUrlUpdate,
      });

      expect(await screen.findByText("Endpoint not found")).toBeInTheDocument();
      expect(screen.queryByTestId("endpoint-info")).not.toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: /back/i }));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate).searchParams.has("endpoint")).toBe(false));
      expect(lastUrlUpdate(onUrlUpdate).searchParams.has("endpoint_tab")).toBe(false);
      expect(await screen.findByTestId("endpoints-table")).toBeInTheDocument();
    });

    it("offers admins the edit view for a dashboard endpoint", async () => {
      renderWithProviders(<PassThroughSettings {...defaultProps} />, { searchParams: "?endpoint=ep-1" });

      expect(await screen.findByTestId("endpoint-info")).toHaveAttribute("data-admin", "true");
    });

    it.each([
      ["a config endpoint without an id", "%2Ffrom-config", { path: "/from-config" }],
      ["a config endpoint with an id", "cfg-1", { id: "cfg-1", path: "/cfg", is_from_config: true }],
    ])("keeps admins read-only on %s, which the dashboard cannot edit", async (_label, param, configFields) => {
      const configEndpoint = { target: "https://config.example.com", headers: {}, ...configFields };
      mockGetEndpoints.mockResolvedValue({ endpoints: [endpoint, configEndpoint] });

      renderWithProviders(<PassThroughSettings {...defaultProps} />, { searchParams: `?endpoint=${param}` });

      const info = await screen.findByTestId("endpoint-info");
      expect(info).toHaveAttribute("data-path", configEndpoint.path);
      expect(info).toHaveAttribute("data-admin", "false");
    });

    it("pushes the opened endpoint to the URL and drops a stale detail tab", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<PassThroughSettings {...defaultProps} />, {
        searchParams: "?endpoint_tab=settings",
        onUrlUpdate,
      });

      await user.click(await screen.findByText("open-ep-1"));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate).searchParams.get("endpoint")).toBe("ep-1"));
      expect(lastUrlUpdate(onUrlUpdate).searchParams.has("endpoint_tab")).toBe(false);
      expect(lastUrlUpdate(onUrlUpdate).options.history).toBe("push");
      expect(screen.getByTestId("endpoint-info")).toHaveTextContent("ep-1");
    });

    it("clears the endpoint and its tab from the URL when the detail closes", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<PassThroughSettings {...defaultProps} />, {
        searchParams: "?endpoint=ep-1&endpoint_tab=settings",
        onUrlUpdate,
      });

      await user.click(await screen.findByRole("button", { name: "close-endpoint" }));

      await waitFor(() => expect(lastUrlUpdate(onUrlUpdate).searchParams.has("endpoint")).toBe(false));
      expect(lastUrlUpdate(onUrlUpdate).searchParams.has("endpoint_tab")).toBe(false);
      expect(lastUrlUpdate(onUrlUpdate).options.history).toBe("push");
      expect(screen.getByTestId("endpoints-table")).toBeInTheDocument();
    });
  });
});
