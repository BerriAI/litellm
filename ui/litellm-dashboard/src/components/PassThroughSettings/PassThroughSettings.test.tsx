import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

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
  default: ({ endpointData }: { endpointData: { id?: string } }) => (
    <div data-testid="endpoint-info">{endpointData.id}</div>
  ),
}));

vi.mock("../molecules/notifications_manager", () => ({
  __esModule: true,
  default: { success: vi.fn(), fromBackend: vi.fn() },
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
    const { container } = render(<PassThroughSettings {...defaultProps} accessToken={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("should hold the table in loading state until the fetch settles", async () => {
    let resolveEndpoints: (value: { endpoints: (typeof endpoint)[] }) => void = () => {};
    mockGetEndpoints.mockReturnValue(
      new Promise((resolve) => {
        resolveEndpoints = resolve;
      }),
    );

    render(<PassThroughSettings {...defaultProps} />);
    expect(screen.getByTestId("endpoints-table")).toHaveAttribute("data-loading", "true");

    await act(async () => {
      resolveEndpoints({ endpoints: [endpoint] });
    });

    await waitFor(() => {
      expect(screen.getByTestId("endpoints-table")).toHaveAttribute("data-loading", "false");
    });
  });

  it("should resolve loading without fetching when the user id is missing", async () => {
    render(<PassThroughSettings {...defaultProps} userID={null} />);

    await waitFor(() => {
      expect(screen.getByTestId("endpoints-table")).toHaveAttribute("data-loading", "false");
    });
    expect(mockGetEndpoints).not.toHaveBeenCalled();
  });

  it("should swap to the endpoint info view when an endpoint is opened", async () => {
    const user = userEvent.setup();
    render(<PassThroughSettings {...defaultProps} />);

    await user.click(await screen.findByText("open-ep-1"));
    expect(screen.getByTestId("endpoint-info")).toHaveTextContent("ep-1");
  });

  it("should confirm before deleting an endpoint", async () => {
    const user = userEvent.setup();
    mockDeleteEndpoint.mockResolvedValue(undefined);
    render(<PassThroughSettings {...defaultProps} />);

    await user.click(await screen.findByText("delete-ep-1"));
    expect(screen.getByText("Delete Pass-Through Endpoint")).toBeInTheDocument();
    expect(mockDeleteEndpoint).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "Delete" }));
    await waitFor(() => {
      expect(mockDeleteEndpoint).toHaveBeenCalledWith("token", "ep-1");
    });
  });
});
