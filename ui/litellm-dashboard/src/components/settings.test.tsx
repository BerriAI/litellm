import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, expect, it, beforeAll, beforeEach, vi } from "vitest";
import Settings from "./settings";
import * as networking from "./networking";

beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => true,
    }),
  });
});

const mockCallbacksData = {
  callbacks: [
    {
      name: "langfuse",
      type: "success",
      variables: {
        LANGFUSE_PUBLIC_KEY: "test_key",
        LANGFUSE_SECRET_KEY: "test_secret",
      },
    },
    {
      name: "datadog",
      type: "success",
      variables: {
        DD_API_KEY: "test_dd_key",
      },
    },
  ],
  available_callbacks: [
    {
      litellm_callback_name: "langfuse",
      ui_callback_name: "Langfuse",
      litellm_callback_params: ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"],
    },
    {
      litellm_callback_name: "datadog",
      ui_callback_name: "Datadog",
      litellm_callback_params: ["DD_API_KEY"],
    },
  ],
  alerts: [],
};

describe("Settings", () => {
  it("should render the settings page", () => {
    vi.spyOn(networking, "alertingSettingsCall").mockResolvedValue([]);
    vi.spyOn(networking, "getEmailEventSettings").mockResolvedValue({ settings: [] });
    vi.spyOn(networking, "getCallbacksCall").mockResolvedValue(mockCallbacksData);

    render(<Settings accessToken="test-token" userRole="admin" userID="test-user" premiumUser={false} />);
  });
});

describe("Logging Callbacks Section", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(networking, "alertingSettingsCall").mockResolvedValue([]);
    vi.spyOn(networking, "getEmailEventSettings").mockResolvedValue({ settings: [] });
  });

  it("should display the list of active callbacks", async () => {
    vi.spyOn(networking, "getCallbacksCall").mockResolvedValue(mockCallbacksData);

    render(<Settings accessToken="test-token" userRole="admin" userID="test-user" premiumUser={false} />);

    await waitFor(
      () => {
        expect(screen.getByText("langfuse")).toBeInTheDocument();
        expect(screen.getByText("datadog")).toBeInTheDocument();
      },
      { timeout: 3000 },
    );
  });

  it("should open add callback modal and display form", async () => {
    vi.spyOn(networking, "getCallbacksCall").mockResolvedValue(mockCallbacksData);

    render(<Settings accessToken="test-token" userRole="admin" userID="test-user" premiumUser={false} />);

    await waitFor(() => {
      expect(screen.getByText("langfuse")).toBeInTheDocument();
    });

    const addButton = screen.getByText("Add Callback");
    fireEvent.click(addButton);

    await waitFor(() => {
      expect(screen.getByText("Add Logging Callback")).toBeInTheDocument();
      expect(screen.getByText("LiteLLM Docs: Logging")).toBeInTheDocument();
    });
  });

  it("should successfully delete a callback", async () => {
    const getCallbacksSpy = vi.spyOn(networking, "getCallbacksCall").mockResolvedValue(mockCallbacksData);
    const deleteCallbackSpy = vi.spyOn(networking, "deleteCallback").mockResolvedValue(undefined);

    const { container } = render(
      <Settings accessToken="test-token" userRole="admin" userID="test-user" premiumUser={false} />,
    );

    await waitFor(() => {
      expect(screen.getByText("langfuse")).toBeInTheDocument();
    });

    const trashIcons = container.querySelectorAll("svg");
    const trashIcon = Array.from(trashIcons).find((svg) => {
      const parentElement = svg.parentElement;
      return parentElement?.className.includes("text-red") || parentElement?.outerHTML.includes("red");
    });

    expect(trashIcon).toBeDefined();
    if (trashIcon && trashIcon.parentElement) {
      fireEvent.click(trashIcon.parentElement);
    }

    await waitFor(() => {
      const modalText = screen.getByText((content, element) => {
        return element?.tagName.toLowerCase() === "p" && content.includes("Are you sure you want to delete");
      });
      expect(modalText).toBeInTheDocument();
    });

    const deleteButton = screen.getByRole("button", { name: "Delete" });
    fireEvent.click(deleteButton);

    await waitFor(() => {
      expect(deleteCallbackSpy).toHaveBeenCalledWith("test-token", "langfuse");
      expect(getCallbacksSpy).toHaveBeenCalledTimes(2);
    });
  });
});
