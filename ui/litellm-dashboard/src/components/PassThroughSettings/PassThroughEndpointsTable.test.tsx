import userEvent from "@testing-library/user-event";
import type { OnUrlUpdateFunction } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";

import { PassThroughEndpointsTable } from "./PassThroughEndpointsTable";
import type { passThroughItem } from "./PassThroughSettings";

const endpoints: passThroughItem[] = [
  {
    id: "ep-1",
    path: "/v1/rerank",
    target: "https://api.cohere.com/v1/rerank",
    headers: { Authorization: "Bearer secret-value" },
    auth: true,
    methods: ["POST"],
  },
  {
    id: "ep-2",
    path: "/bria",
    target: "https://engine.prod.bria-api.com",
    headers: {},
    auth: false,
  },
];

const defaultProps = {
  endpoints,
  isLoading: false,
  onEndpointClick: vi.fn(),
  onDeleteClick: vi.fn(),
};

describe("PassThroughEndpointsTable", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should render a row per endpoint with path and target", () => {
    renderWithProviders(<PassThroughEndpointsTable {...defaultProps} />);
    expect(screen.getByText("/v1/rerank")).toBeInTheDocument();
    expect(screen.getByText("https://api.cohere.com/v1/rerank")).toBeInTheDocument();
    expect(screen.getByText("/bria")).toBeInTheDocument();
  });

  it("should open the endpoint when its ID is clicked", async () => {
    const user = userEvent.setup();
    const onEndpointClick = vi.fn();
    renderWithProviders(<PassThroughEndpointsTable {...defaultProps} onEndpointClick={onEndpointClick} />);
    await user.click(screen.getByRole("button", { name: "ep-1" }));
    expect(onEndpointClick).toHaveBeenCalledWith("ep-1");
  });

  it("should show method chips, or ALL when no methods are set", () => {
    renderWithProviders(<PassThroughEndpointsTable {...defaultProps} />);
    expect(screen.getByText("POST")).toBeInTheDocument();
    expect(screen.getByText("ALL")).toBeInTheDocument();
  });

  it("should show authentication as Yes or No", () => {
    renderWithProviders(<PassThroughEndpointsTable {...defaultProps} />);
    expect(screen.getByText("Yes")).toBeInTheDocument();
    expect(screen.getByText("No")).toBeInTheDocument();
  });

  it("should mask headers until the visibility toggle is clicked", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PassThroughEndpointsTable {...defaultProps} />);

    expect(screen.queryByText(/secret-value/)).not.toBeInTheDocument();
    const toggles = screen.getAllByRole("button", { name: "Show headers" });
    await user.click(toggles[0]);
    expect(screen.getByText(/secret-value/)).toBeInTheDocument();
  });

  it("should edit and delete an endpoint through the actions menu", async () => {
    const user = userEvent.setup();
    const onEndpointClick = vi.fn();
    const onDeleteClick = vi.fn();
    renderWithProviders(
      <PassThroughEndpointsTable {...defaultProps} onEndpointClick={onEndpointClick} onDeleteClick={onDeleteClick} />,
    );

    await user.click(screen.getByTestId("endpoint-actions-ep-1"));
    await user.click(await screen.findByTestId("endpoint-action-edit"));
    expect(onEndpointClick).toHaveBeenCalledWith("ep-1");

    await user.click(screen.getByTestId("endpoint-actions-ep-1"));
    await user.click(await screen.findByTestId("endpoint-action-delete"));
    expect(onDeleteClick).toHaveBeenCalledWith("ep-1");
  });

  it("should disable edit and delete for config-defined endpoints", async () => {
    const user = userEvent.setup();
    const onEndpointClick = vi.fn();
    const onDeleteClick = vi.fn();
    const configEndpoint: passThroughItem = {
      id: "ep-config",
      path: "/from-config",
      target: "https://config.example.com",
      headers: {},
      is_from_config: true,
    };
    renderWithProviders(
      <PassThroughEndpointsTable
        {...defaultProps}
        endpoints={[configEndpoint]}
        onEndpointClick={onEndpointClick}
        onDeleteClick={onDeleteClick}
      />,
    );

    await user.click(screen.getByTestId("endpoint-actions-ep-config"));
    const editItem = await screen.findByTestId("endpoint-action-edit");
    const deleteItem = await screen.findByTestId("endpoint-action-delete");

    expect(editItem).toHaveAttribute("data-disabled");
    expect(deleteItem).toHaveAttribute("data-disabled");
    expect(screen.getByTestId("endpoint-config-hint")).toHaveTextContent(
      "This endpoint is defined in the config file and cannot be edited or deleted on the dashboard.",
    );

    await user.click(editItem);
    await user.click(deleteItem);

    expect(onEndpointClick).not.toHaveBeenCalled();
    expect(onDeleteClick).not.toHaveBeenCalled();
  });

  it("should not show the config hint for DB endpoints", async () => {
    const user = userEvent.setup();
    renderWithProviders(<PassThroughEndpointsTable {...defaultProps} />);

    await user.click(screen.getByTestId("endpoint-actions-ep-1"));
    await screen.findByTestId("endpoint-action-delete");
    expect(screen.queryByTestId("endpoint-config-hint")).not.toBeInTheDocument();
  });

  it("should label endpoint source as Config or DB", () => {
    const configEndpoint: passThroughItem = {
      id: "ep-config",
      path: "/from-config",
      target: "https://config.example.com",
      headers: {},
      is_from_config: true,
    };
    renderWithProviders(<PassThroughEndpointsTable {...defaultProps} endpoints={[...endpoints, configEndpoint]} />);
    expect(screen.getByText("Config")).toBeInTheDocument();
    expect(screen.getAllByText("DB")).toHaveLength(2);
  });

  it("should disable edit and delete for endpoints without an id", async () => {
    const user = userEvent.setup();
    const onEndpointClick = vi.fn();
    const onDeleteClick = vi.fn();
    const endpointWithoutId: passThroughItem = { path: "/legacy", target: "https://legacy.example.com", headers: {} };
    renderWithProviders(
      <PassThroughEndpointsTable
        {...defaultProps}
        endpoints={[endpointWithoutId]}
        onEndpointClick={onEndpointClick}
        onDeleteClick={onDeleteClick}
      />,
    );

    await user.click(screen.getByTestId("endpoint-actions-/legacy"));
    const editItem = await screen.findByTestId("endpoint-action-edit");
    const deleteItem = await screen.findByTestId("endpoint-action-delete");

    expect(editItem).toHaveAttribute("data-disabled");
    expect(deleteItem).toHaveAttribute("data-disabled");

    await user.click(editItem);
    await user.click(deleteItem);

    expect(onEndpointClick).not.toHaveBeenCalled();
    expect(onDeleteClick).not.toHaveBeenCalled();
  });

  it("should show the empty state when there are no endpoints", () => {
    renderWithProviders(<PassThroughEndpointsTable {...defaultProps} endpoints={[]} />);
    expect(screen.getByText("No pass-through endpoints configured")).toBeInTheDocument();
  });

  it("should show skeleton rows while loading", () => {
    renderWithProviders(<PassThroughEndpointsTable {...defaultProps} endpoints={[]} isLoading />);
    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
    expect(screen.queryByText("No pass-through endpoints configured")).not.toBeInTheDocument();
  });

  describe("URL page state", () => {
    const manyEndpoints: passThroughItem[] = Array.from({ length: 27 }, (_, index) => ({
      id: `ep-${index + 1}`,
      path: `/route-${index + 1}`,
      target: `https://upstream.example.com/${index + 1}`,
      headers: {},
    }));

    it("keeps the page named in the URL once the endpoints finish loading", () => {
      const { rerender } = renderWithProviders(
        <PassThroughEndpointsTable {...defaultProps} endpoints={[]} isLoading />,
        { searchParams: "?pass_through_page=2" },
      );

      rerender(<PassThroughEndpointsTable {...defaultProps} endpoints={manyEndpoints} isLoading={false} />);

      expect(screen.getByText("/route-26")).toBeInTheDocument();
      expect(screen.getByText("/route-27")).toBeInTheDocument();
      expect(screen.queryByText("/route-1")).not.toBeInTheDocument();
      expect(screen.getByTestId("pagination-page")).toHaveTextContent("Page 2 of 2");
    });

    it("writes a page change to the URL", async () => {
      const user = userEvent.setup();
      const onUrlUpdate = vi.fn<OnUrlUpdateFunction>();
      renderWithProviders(<PassThroughEndpointsTable {...defaultProps} endpoints={manyEndpoints} />, { onUrlUpdate });

      await user.click(screen.getByTestId("pagination-next"));

      await waitFor(() => expect(onUrlUpdate.mock.calls.at(-1)?.[0].searchParams.get("pass_through_page")).toBe("2"));
      expect(screen.getByText("/route-26")).toBeInTheDocument();
    });
  });
});
