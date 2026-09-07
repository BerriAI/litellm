import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

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
    render(<PassThroughEndpointsTable {...defaultProps} />);
    expect(screen.getByText("/v1/rerank")).toBeInTheDocument();
    expect(screen.getByText("https://api.cohere.com/v1/rerank")).toBeInTheDocument();
    expect(screen.getByText("/bria")).toBeInTheDocument();
  });

  it("should open the endpoint when its ID is clicked", async () => {
    const user = userEvent.setup();
    const onEndpointClick = vi.fn();
    render(<PassThroughEndpointsTable {...defaultProps} onEndpointClick={onEndpointClick} />);
    await user.click(screen.getByRole("button", { name: "ep-1" }));
    expect(onEndpointClick).toHaveBeenCalledWith("ep-1");
  });

  it("should show method chips, or ALL when no methods are set", () => {
    render(<PassThroughEndpointsTable {...defaultProps} />);
    expect(screen.getByText("POST")).toBeInTheDocument();
    expect(screen.getByText("ALL")).toBeInTheDocument();
  });

  it("should show authentication as Yes or No", () => {
    render(<PassThroughEndpointsTable {...defaultProps} />);
    expect(screen.getByText("Yes")).toBeInTheDocument();
    expect(screen.getByText("No")).toBeInTheDocument();
  });

  it("should mask headers until the visibility toggle is clicked", async () => {
    const user = userEvent.setup();
    render(<PassThroughEndpointsTable {...defaultProps} />);

    expect(screen.queryByText(/secret-value/)).not.toBeInTheDocument();
    const toggles = screen.getAllByRole("button", { name: "Show headers" });
    await user.click(toggles[0]);
    expect(screen.getByText(/secret-value/)).toBeInTheDocument();
  });

  it("should edit and delete an endpoint through the actions menu", async () => {
    const user = userEvent.setup();
    const onEndpointClick = vi.fn();
    const onDeleteClick = vi.fn();
    render(
      <PassThroughEndpointsTable {...defaultProps} onEndpointClick={onEndpointClick} onDeleteClick={onDeleteClick} />,
    );

    await user.click(screen.getByTestId("endpoint-actions-ep-1"));
    await user.click(await screen.findByTestId("endpoint-action-edit"));
    expect(onEndpointClick).toHaveBeenCalledWith("ep-1");

    await user.click(screen.getByTestId("endpoint-actions-ep-1"));
    await user.click(await screen.findByTestId("endpoint-action-delete"));
    expect(onDeleteClick).toHaveBeenCalledWith("ep-1");
  });

  it("should disable edit and delete for endpoints without an id", async () => {
    const user = userEvent.setup();
    const onEndpointClick = vi.fn();
    const onDeleteClick = vi.fn();
    const endpointWithoutId: passThroughItem = { path: "/legacy", target: "https://legacy.example.com", headers: {} };
    render(
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
    render(<PassThroughEndpointsTable {...defaultProps} endpoints={[]} />);
    expect(screen.getByText("No pass-through endpoints configured")).toBeInTheDocument();
  });

  it("should show skeleton rows while loading", () => {
    render(<PassThroughEndpointsTable {...defaultProps} endpoints={[]} isLoading />);
    expect(screen.getAllByTestId("skeleton-row").length).toBeGreaterThan(0);
    expect(screen.queryByText("No pass-through endpoints configured")).not.toBeInTheDocument();
  });
});
