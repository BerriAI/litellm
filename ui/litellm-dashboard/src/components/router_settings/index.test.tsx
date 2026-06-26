import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderWithProviders, screen, waitFor } from "../../../tests/test-utils";
import userEvent from "@testing-library/user-event";
import RouterSettings from "./index";

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>();
  return {
    ...actual,
    Select: Object.assign(
      ({ value, onChange, children }: any) => (
        <select data-testid="strategy-select" value={value ?? ""} onChange={(e) => onChange(e.target.value)}>
          {children}
        </select>
      ),
      {
        Option: ({ value, children }: any) => <option value={value}>{children}</option>,
      },
    ),
  };
});

vi.mock("@/components/networking", () => ({
  getCallbacksCall: vi.fn(),
  getRouterSettingsCall: vi.fn(),
  setCallbacksCall: vi.fn(),
}));

import { getCallbacksCall, getRouterSettingsCall, setCallbacksCall } from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";

const mockCallbacksResponse = {
  router_settings: {
    routing_strategy: "simple-shuffle",
    num_retries: 3,
    timeout: 30,
  },
};

const mockRouterSettingsResponse = {
  fields: [
    {
      field_name: "routing_strategy",
      ui_field_name: "Routing Strategy",
      field_description: "How requests are distributed",
      options: ["simple-shuffle", "latency-based-routing"],
      link: null,
    },
    {
      field_name: "enable_tag_filtering",
      ui_field_name: "Tag Filtering",
      field_description: "Route by tag",
      field_value: false,
      link: null,
    },
  ],
  routing_strategy_descriptions: {
    "simple-shuffle": "Randomly pick a deployment",
    "latency-based-routing": "Pick the lowest-latency deployment",
  },
};

const defaultProps = {
  accessToken: "test-token",
  userRole: "Admin",
  userID: "user-1",
  modelData: null,
};

describe("RouterSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(getCallbacksCall).mockResolvedValue(mockCallbacksResponse);
    vi.mocked(getRouterSettingsCall).mockResolvedValue(mockRouterSettingsResponse);
    vi.mocked(setCallbacksCall).mockResolvedValue({});
  });

  it("should render nothing when accessToken is null", () => {
    const { container } = renderWithProviders(<RouterSettings {...defaultProps} accessToken={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("should render the Save Changes and Reset buttons when authenticated", () => {
    renderWithProviders(<RouterSettings {...defaultProps} />);
    expect(screen.getByRole("button", { name: /save changes/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /reset/i })).toBeInTheDocument();
  });

  it("should fetch callbacks and router settings on mount", async () => {
    renderWithProviders(<RouterSettings {...defaultProps} />);

    await waitFor(() => {
      expect(getCallbacksCall).toHaveBeenCalledWith("test-token", "user-1", "Admin");
    });
    expect(getRouterSettingsCall).toHaveBeenCalledWith("test-token");
  });

  it("should not fetch data when any required prop is missing", () => {
    renderWithProviders(<RouterSettings {...defaultProps} userRole={null} />);
    expect(getCallbacksCall).not.toHaveBeenCalled();
  });

  it("should render routing strategies loaded from the API", async () => {
    renderWithProviders(<RouterSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByTestId("strategy-select")).toBeInTheDocument();
    });

    const select = screen.getByTestId("strategy-select") as HTMLSelectElement;
    const optionValues = Array.from(select.options).map((o) => o.value);
    expect(optionValues).toContain("simple-shuffle");
    expect(optionValues).toContain("latency-based-routing");
  });

  it("should call setCallbacksCall with updated settings on Save Changes", async () => {
    const user = userEvent.setup();
    renderWithProviders(<RouterSettings {...defaultProps} />);

    // Wait for the strategy select to appear — it only renders after getRouterSettingsCall resolves
    await waitFor(() => {
      expect(screen.getByTestId("strategy-select")).toBeInTheDocument();
    });

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    expect(setCallbacksCall).toHaveBeenCalledWith(
      "test-token",
      expect.objectContaining({
        router_settings: expect.objectContaining({
          routing_strategy: "simple-shuffle",
        }),
      }),
    );
  });

  it("should send routing_groups as a parsed list, not a stringified one", async () => {
    vi.mocked(getCallbacksCall).mockResolvedValue({
      router_settings: {
        routing_strategy: "simple-shuffle",
        routing_groups: [],
      },
    });

    const user = userEvent.setup();
    renderWithProviders(<RouterSettings {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByTestId("strategy-select")).toBeInTheDocument();
    });

    const input = document.querySelector('input[name="routing_groups"]') as HTMLInputElement | null;
    expect(input?.value).toBe("[]");

    await user.click(screen.getByRole("button", { name: /save changes/i }));

    const payload = vi.mocked(setCallbacksCall).mock.calls[0][1] as {
      router_settings: { routing_groups: unknown };
    };
    expect(payload.router_settings.routing_groups).toEqual([]);
    expect(typeof payload.router_settings.routing_groups).not.toBe("string");
  });

  it("should show a success notification after saving", async () => {
    const user = userEvent.setup();
    renderWithProviders(<RouterSettings {...defaultProps} />);

    // Wait for data to load before interacting
    await waitFor(() => {
      expect(screen.getByTestId("strategy-select")).toBeInTheDocument();
    });
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    expect(NotificationsManager.success).toHaveBeenCalledWith("router settings updated successfully");
  });
});
