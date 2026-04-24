import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { VectorStore } from "./types";
import VectorStoreSelector from "./VectorStoreSelector";

const mockVectorStoreListCall = vi.fn();

vi.mock("../networking", () => ({
  vectorStoreListCall: (...args: any[]) => mockVectorStoreListCall(...args),
}));

const mockOnChange = vi.fn();
const mockAccessToken = "test-token";

const mockVectorStores: VectorStore[] = [
  {
    vector_store_id: "store-1",
    custom_llm_provider: "openai",
    vector_store_name: "My Store",
    vector_store_description: "A test store",
    created_at: "2024-01-01T00:00:00Z",
    updated_at: "2024-01-01T00:00:00Z",
  },
  {
    vector_store_id: "store-2",
    custom_llm_provider: "azure",
    vector_store_name: "Another Store",
    vector_store_description: "Another test store",
    created_at: "2024-01-02T00:00:00Z",
    updated_at: "2024-01-02T00:00:00Z",
  },
  {
    vector_store_id: "store-3",
    custom_llm_provider: "pg_vector",
    vector_store_description: "Store without name",
    created_at: "2024-01-03T00:00:00Z",
    updated_at: "2024-01-03T00:00:00Z",
  },
];

const defaultProps = {
  onChange: mockOnChange,
  accessToken: mockAccessToken,
};

const renderComponent = (props: Partial<React.ComponentProps<typeof VectorStoreSelector>> = {}) => {
  return render(<VectorStoreSelector {...defaultProps} {...props} />);
};

const waitForDataFetch = async () => {
  await waitFor(() => {
    expect(mockVectorStoreListCall).toHaveBeenCalled();
  });
};

const openPopover = async () => {
  // The trigger is the first button rendered — chip Remove buttons come after it.
  const trigger = screen.getAllByRole("button")[0];
  await act(async () => {
    fireEvent.click(trigger);
  });
  return trigger;
};

describe("VectorStoreSelector", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockVectorStoreListCall.mockResolvedValue({
      data: mockVectorStores,
    });
  });

  describe("Rendering", () => {
    it("should render the trigger button", () => {
      renderComponent();
      expect(screen.getByRole("button")).toBeInTheDocument();
    });

    it("should render with default placeholder", async () => {
      renderComponent();
      await waitForDataFetch();
      expect(screen.getByText("Select vector stores")).toBeInTheDocument();
    });

    it("should render with custom placeholder", async () => {
      renderComponent({ placeholder: "Choose stores" });
      await waitForDataFetch();
      expect(screen.getByText("Choose stores")).toBeInTheDocument();
    });

    it("should apply custom className", () => {
      renderComponent({ className: "custom-class" });
      const trigger = screen.getByRole("button");
      expect(trigger).toHaveClass("custom-class");
    });

    it("should render as disabled when disabled prop set", () => {
      renderComponent({ disabled: true });
      const trigger = screen.getByRole("button");
      expect(trigger).toBeDisabled();
    });

    it("should render as enabled by default", () => {
      renderComponent();
      const trigger = screen.getByRole("button");
      expect(trigger).not.toBeDisabled();
    });
  });

  describe("Data fetching", () => {
    it("should fetch vector stores on mount when accessToken is provided", async () => {
      renderComponent();
      await waitFor(() => {
        expect(mockVectorStoreListCall).toHaveBeenCalledWith(mockAccessToken);
      });
    });

    it("should not fetch vector stores when accessToken is falsy", () => {
      const { rerender } = render(<VectorStoreSelector {...defaultProps} accessToken="" />);
      expect(mockVectorStoreListCall).not.toHaveBeenCalled();

      rerender(<VectorStoreSelector {...defaultProps} accessToken={null as any} />);
      expect(mockVectorStoreListCall).not.toHaveBeenCalled();

      rerender(<VectorStoreSelector {...defaultProps} accessToken={undefined as any} />);
      expect(mockVectorStoreListCall).not.toHaveBeenCalled();
    });

    it("should fetch vector stores again when accessToken changes", async () => {
      const { rerender } = render(<VectorStoreSelector {...defaultProps} accessToken="token-1" />);
      await waitFor(() => {
        expect(mockVectorStoreListCall).toHaveBeenCalledWith("token-1");
      });

      vi.clearAllMocks();
      rerender(<VectorStoreSelector {...defaultProps} accessToken="token-2" />);
      await waitFor(() => {
        expect(mockVectorStoreListCall).toHaveBeenCalledWith("token-2");
      });
    });

    it("should show loading placeholder while fetching", async () => {
      let resolvePromise: (value: any) => void;
      const promise = new Promise((resolve) => {
        resolvePromise = resolve;
      });
      mockVectorStoreListCall.mockReturnValue(promise);

      renderComponent();
      expect(screen.getByText(/loading/i)).toBeInTheDocument();

      resolvePromise!({ data: mockVectorStores });
      await waitForDataFetch();
      await waitFor(() => {
        expect(screen.getByText("Select vector stores")).toBeInTheDocument();
      });
    });

    it("should clear loading state after failed fetch", async () => {
      const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
      mockVectorStoreListCall.mockRejectedValueOnce(new Error("Network error"));

      renderComponent();
      await waitForDataFetch();

      await waitFor(() => {
        expect(screen.getByText("Select vector stores")).toBeInTheDocument();
      });
      consoleErrorSpy.mockRestore();
    });
  });

  describe("Options rendering", () => {
    it("should render vector store options when opened", async () => {
      renderComponent();
      await waitForDataFetch();
      await openPopover();

      expect(screen.getByText("My Store (store-1)")).toBeInTheDocument();
      expect(screen.getByText("Another Store (store-2)")).toBeInTheDocument();
      expect(screen.getByText("store-3 (store-3)")).toBeInTheDocument();
    });

    it("should fall back to vector_store_id for label when name is missing", async () => {
      renderComponent();
      await waitForDataFetch();
      await openPopover();

      expect(screen.getByText("store-3 (store-3)")).toBeInTheDocument();
    });

    it("should use description as title attribute when available", async () => {
      renderComponent();
      await waitForDataFetch();
      await openPopover();

      const option1 = screen.getByText("My Store (store-1)");
      expect(option1).toHaveAttribute("title", "A test store");
    });

    it("should fall back to vector_store_id as title when description is missing", async () => {
      const stores: VectorStore[] = [
        {
          vector_store_id: "store-no-desc",
          custom_llm_provider: "openai",
          created_at: "2024-01-01T00:00:00Z",
          updated_at: "2024-01-01T00:00:00Z",
        },
      ];
      mockVectorStoreListCall.mockResolvedValueOnce({ data: stores });

      renderComponent();
      await waitForDataFetch();
      await openPopover();

      const option = screen.getByText("store-no-desc (store-no-desc)");
      expect(option).toHaveAttribute("title", "store-no-desc");
    });

    it("should show 'No matches' when no options remain", async () => {
      mockVectorStoreListCall.mockResolvedValueOnce({ data: [] });

      renderComponent();
      await waitForDataFetch();
      await openPopover();

      expect(screen.getByText("No matches")).toBeInTheDocument();
    });

    it("should handle response without data property", async () => {
      mockVectorStoreListCall.mockResolvedValueOnce({});

      renderComponent();
      await waitForDataFetch();
      await openPopover();

      expect(screen.getByText("No matches")).toBeInTheDocument();
    });
  });

  describe("Value prop", () => {
    it("should render selected values as chips", async () => {
      renderComponent({ value: ["store-1", "store-2"] });
      await waitForDataFetch();

      await waitFor(() => {
        expect(screen.getByText("My Store")).toBeInTheDocument();
        expect(screen.getByText("Another Store")).toBeInTheDocument();
      });
    });

    it("should render placeholder when value is empty", async () => {
      renderComponent({ value: [] });
      await waitForDataFetch();
      expect(screen.getByText("Select vector stores")).toBeInTheDocument();
    });

    it("should render placeholder when value is undefined", async () => {
      renderComponent({ value: undefined });
      await waitForDataFetch();
      expect(screen.getByText("Select vector stores")).toBeInTheDocument();
    });

    it("should exclude selected stores from the option list", async () => {
      renderComponent({ value: ["store-1"] });
      await waitForDataFetch();
      await openPopover();

      expect(screen.queryByText("My Store (store-1)")).not.toBeInTheDocument();
      expect(screen.getByText("Another Store (store-2)")).toBeInTheDocument();
    });
  });

  describe("onChange callback", () => {
    it("should call onChange when an option is selected", async () => {
      renderComponent();
      await waitForDataFetch();
      await openPopover();

      const option = screen.getByText("My Store (store-1)");
      await act(async () => {
        fireEvent.click(option);
      });

      expect(mockOnChange).toHaveBeenCalledWith(["store-1"]);
    });

    it("should append to existing selection on subsequent select", async () => {
      renderComponent({ value: ["store-1"] });
      await waitForDataFetch();
      await openPopover();

      const option = screen.getByText("Another Store (store-2)");
      await act(async () => {
        fireEvent.click(option);
      });

      expect(mockOnChange).toHaveBeenCalledWith(["store-1", "store-2"]);
    });

    it("should call onChange when removing a selected chip", async () => {
      renderComponent({ value: ["store-1", "store-2"] });
      await waitForDataFetch();

      const removeBtn = screen.getByLabelText("Remove My Store");
      await act(async () => {
        fireEvent.click(removeBtn);
      });

      expect(mockOnChange).toHaveBeenCalledWith(["store-2"]);
    });
  });

  describe("Filtering", () => {
    it("should filter options by search query", async () => {
      renderComponent();
      await waitForDataFetch();
      await openPopover();

      const search = screen.getByPlaceholderText(/search vector stores/i);
      await act(async () => {
        fireEvent.change(search, { target: { value: "Another" } });
      });

      expect(screen.queryByText("My Store (store-1)")).not.toBeInTheDocument();
      expect(screen.getByText("Another Store (store-2)")).toBeInTheDocument();
    });
  });

  describe("Error handling", () => {
    it("should handle fetch errors gracefully", async () => {
      const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
      const error = new Error("Network error");
      mockVectorStoreListCall.mockRejectedValueOnce(error);

      renderComponent();
      await waitForDataFetch();

      expect(consoleErrorSpy).toHaveBeenCalledWith("Error fetching vector stores:", error);
      consoleErrorSpy.mockRestore();
    });

    it("should not crash when fetch throws non-Error", async () => {
      const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
      mockVectorStoreListCall.mockRejectedValueOnce("String error");

      renderComponent();
      await waitForDataFetch();

      expect(consoleErrorSpy).toHaveBeenCalledWith("Error fetching vector stores:", "String error");
      consoleErrorSpy.mockRestore();
    });

    it("should continue to render after error", async () => {
      const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
      mockVectorStoreListCall.mockRejectedValueOnce(new Error("Network error"));

      renderComponent();
      await waitForDataFetch();

      expect(screen.getByRole("button")).toBeInTheDocument();
      consoleErrorSpy.mockRestore();
    });
  });

  describe("Edge cases", () => {
    it("should render minimal vector stores with only id", async () => {
      const minimalStores: VectorStore[] = [
        {
          vector_store_id: "minimal-store",
          custom_llm_provider: "openai",
          created_at: "2024-01-01T00:00:00Z",
          updated_at: "2024-01-01T00:00:00Z",
        },
      ];
      mockVectorStoreListCall.mockResolvedValueOnce({ data: minimalStores });

      renderComponent();
      await waitForDataFetch();
      await openPopover();

      expect(screen.getByText("minimal-store (minimal-store)")).toBeInTheDocument();
    });

    it("should render very long vector store names", async () => {
      const longName = "A".repeat(200);
      const longNameStores: VectorStore[] = [
        {
          vector_store_id: "store-long",
          custom_llm_provider: "openai",
          vector_store_name: longName,
          created_at: "2024-01-01T00:00:00Z",
          updated_at: "2024-01-01T00:00:00Z",
        },
      ];
      mockVectorStoreListCall.mockResolvedValueOnce({ data: longNameStores });

      renderComponent();
      await waitForDataFetch();
      await openPopover();

      expect(screen.getByText(`${longName} (store-long)`)).toBeInTheDocument();
    });

    it("should render special characters in names", async () => {
      const specialCharStores: VectorStore[] = [
        {
          vector_store_id: "store-special",
          custom_llm_provider: "openai",
          vector_store_name: 'Store & Co. <Test> "Quotes"',
          created_at: "2024-01-01T00:00:00Z",
          updated_at: "2024-01-01T00:00:00Z",
        },
      ];
      mockVectorStoreListCall.mockResolvedValueOnce({ data: specialCharStores });

      renderComponent();
      await waitForDataFetch();
      await openPopover();

      expect(screen.getByText(/Store & Co\. <Test> "Quotes"/)).toBeInTheDocument();
    });
  });
});
