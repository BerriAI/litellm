import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { VectorStore } from "./types";
import VectorStoreSelector from "./VectorStoreSelector";

const mockVectorStoreListCall = vi.fn();

vi.mock("../networking", () => ({
  vectorStoreListCall: (...args: unknown[]) => mockVectorStoreListCall(...args),
}));

vi.mock("@/components/shared/MultiSelect", () => ({
  MultiSelect: vi.fn(),
}));

import { MultiSelect as MockedMultiSelect } from "@/components/shared/MultiSelect";

(MockedMultiSelect as unknown as ReturnType<typeof vi.fn>).mockImplementation(
  (props: {
    onValueChange?: (value: string[]) => void;
    value?: string[];
    placeholder?: string;
    loading?: boolean;
    className?: string;
    disabled?: boolean;
    options?: Array<{ value: string; label: string; description?: string }>;
  }) => {
    const { onValueChange, value, placeholder, loading, className, disabled, options } = props;

    return (
      <div
        data-testid="vector-store-select"
        data-loading={loading}
        data-disabled={disabled}
        data-placeholder={placeholder}
        data-value={value !== undefined ? JSON.stringify(value) : undefined}
        data-options={JSON.stringify(options)}
        className={className}
        onClick={(e) => {
          const testSelection = (e.target as HTMLElement).getAttribute("data-test-selection");
          if (testSelection && onValueChange) {
            onValueChange(JSON.parse(testSelection) as string[]);
          } else if (onValueChange && options && options.length > 0) {
            onValueChange([options[0].value]);
          }
        }}
      >
        {options?.map((opt) => (
          <div
            key={opt.value}
            data-option-value={opt.value}
            data-option-label={opt.label}
            data-option-description={opt.description}
            data-testid={`option-${opt.value}`}
          >
            {opt.label}
          </div>
        ))}
      </div>
    );
  },
);

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

const renderComponent = (props = {}) => {
  return render(<VectorStoreSelector {...defaultProps} {...props} />);
};

const waitForDataFetch = async () => {
  await waitFor(() => {
    expect(mockVectorStoreListCall).toHaveBeenCalled();
  });
};

const getSelectElement = () => screen.getByTestId("vector-store-select");

const getOptionElements = () =>
  screen.queryAllByTestId(/^option-/).filter((el) => el.hasAttribute("data-option-value"));

describe("VectorStoreSelector", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockVectorStoreListCall.mockResolvedValue({
      data: mockVectorStores,
    });
  });

  describe("Rendering", () => {
    it("should render the select component", () => {
      renderComponent();
      expect(getSelectElement()).toBeInTheDocument();
    });

    it("should render with default placeholder", () => {
      renderComponent();
      expect(getSelectElement()).toHaveAttribute("data-placeholder", "Select vector stores");
    });

    it("should render with custom placeholder", () => {
      renderComponent({ placeholder: "Choose stores" });
      expect(getSelectElement()).toHaveAttribute("data-placeholder", "Choose stores");
    });

    it("should apply custom className", () => {
      renderComponent({ className: "custom-class" });
      expect(getSelectElement()).toHaveClass("custom-class");
    });

    it("should render with disabled state", () => {
      renderComponent({ disabled: true });
      expect(getSelectElement()).toHaveAttribute("data-disabled", "true");
    });

    it("should render with enabled state by default", () => {
      renderComponent();
      expect(getSelectElement()).toHaveAttribute("data-disabled", "false");
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

      rerender(<VectorStoreSelector {...defaultProps} accessToken={null as unknown as string} />);
      expect(mockVectorStoreListCall).not.toHaveBeenCalled();

      rerender(<VectorStoreSelector {...defaultProps} accessToken={undefined as unknown as string} />);
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

    it("should set loading state while fetching", async () => {
      let resolvePromise: (value: unknown) => void;
      const promise = new Promise((resolve) => {
        resolvePromise = resolve;
      });
      mockVectorStoreListCall.mockReturnValue(promise);

      renderComponent();
      const select = getSelectElement();
      expect(select).toHaveAttribute("data-loading", "true");

      resolvePromise!({ data: mockVectorStores });
      await waitFor(() => {
        expect(select).toHaveAttribute("data-loading", "false");
      });
    });

    it("should clear loading state after successful fetch", async () => {
      renderComponent();
      await waitForDataFetch();
      expect(getSelectElement()).toHaveAttribute("data-loading", "false");
    });

    it("should clear loading state after failed fetch", async () => {
      const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
      mockVectorStoreListCall.mockRejectedValueOnce(new Error("Network error"));

      renderComponent();
      await waitForDataFetch();

      expect(getSelectElement()).toHaveAttribute("data-loading", "false");
      consoleErrorSpy.mockRestore();
    });
  });

  describe("Options rendering", () => {
    it("should render vector store options after successful fetch", async () => {
      renderComponent();
      await waitForDataFetch();

      expect(screen.getByText("My Store (store-1)")).toBeInTheDocument();
      expect(screen.getByText("Another Store (store-2)")).toBeInTheDocument();
      expect(screen.getByText("store-3 (store-3)")).toBeInTheDocument();
    });

    it("should use vector_store_name when available for label", async () => {
      renderComponent();
      await waitForDataFetch();

      const option1 = screen.getByText("My Store (store-1)");
      expect(option1).toBeInTheDocument();
      expect(option1).toHaveAttribute("data-option-description", "A test store");
    });

    it("should fallback to vector_store_id when vector_store_name is missing", async () => {
      renderComponent();
      await waitForDataFetch();

      const option3 = screen.getByText("store-3 (store-3)");
      expect(option3).toBeInTheDocument();
      expect(option3).toHaveAttribute("data-option-description", "Store without name");
    });

    it("should use vector_store_description as description when available", async () => {
      renderComponent();
      await waitForDataFetch();

      const option1 = screen.getByText("My Store (store-1)");
      expect(option1).toHaveAttribute("data-option-description", "A test store");
    });

    it("should omit description when vector_store_description is missing", async () => {
      const storesWithoutDescription: VectorStore[] = [
        {
          vector_store_id: "store-no-desc",
          custom_llm_provider: "openai",
          created_at: "2024-01-01T00:00:00Z",
          updated_at: "2024-01-01T00:00:00Z",
        },
      ];
      mockVectorStoreListCall.mockResolvedValueOnce({
        data: storesWithoutDescription,
      });

      renderComponent();
      await waitForDataFetch();

      const option = screen.getByText("store-no-desc (store-no-desc)");
      expect(option).not.toHaveAttribute("data-option-description");
    });

    it("should use vector_store_id as option value", async () => {
      renderComponent();
      await waitForDataFetch();

      const option1 = screen.getByText("My Store (store-1)");
      expect(option1).toHaveAttribute("data-option-value", "store-1");
    });

    it("should handle empty vector stores array", async () => {
      mockVectorStoreListCall.mockResolvedValueOnce({
        data: [],
      });

      renderComponent();
      await waitForDataFetch();

      expect(getOptionElements().length).toBe(0);
    });

    it("should handle response without data property", async () => {
      mockVectorStoreListCall.mockResolvedValueOnce({});

      renderComponent();
      await waitForDataFetch();

      expect(getOptionElements().length).toBe(0);
    });
  });

  describe("Value prop", () => {
    it("should set initial value when value prop is provided", async () => {
      renderComponent({ value: ["store-1", "store-2"] });
      await waitForDataFetch();

      expect(getSelectElement()).toHaveAttribute("data-value", JSON.stringify(["store-1", "store-2"]));
    });

    it("should handle empty value array", async () => {
      renderComponent({ value: [] });
      await waitForDataFetch();

      expect(getSelectElement()).toHaveAttribute("data-value", JSON.stringify([]));
    });

    it("should handle undefined value", async () => {
      renderComponent({ value: undefined });
      await waitForDataFetch();

      expect(getSelectElement()).not.toHaveAttribute("data-value");
    });
  });

  describe("onChange callback", () => {
    it("should call onChange when selection changes", async () => {
      renderComponent();
      await waitForDataFetch();

      const select = getSelectElement();
      select.setAttribute("data-test-selection", '["store-1"]');
      fireEvent.click(select);

      expect(mockOnChange).toHaveBeenCalledWith(["store-1"]);
    });

    it("should call onChange with multiple selected values", async () => {
      renderComponent();
      await waitForDataFetch();

      const select = getSelectElement();
      select.setAttribute("data-test-selection", '["store-1", "store-2"]');
      fireEvent.click(select);

      expect(mockOnChange).toHaveBeenCalledWith(["store-1", "store-2"]);
    });

    it("should call onChange when deselecting options", async () => {
      renderComponent({ value: ["store-1", "store-2"] });
      await waitForDataFetch();

      const select = getSelectElement();
      select.setAttribute("data-test-selection", '["store-2"]');
      fireEvent.click(select);

      expect(mockOnChange).toHaveBeenCalledWith(["store-2"]);
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

    it("should continue to work after error", async () => {
      const consoleErrorSpy = vi.spyOn(console, "error").mockImplementation(() => {});
      mockVectorStoreListCall.mockRejectedValueOnce(new Error("Network error"));

      renderComponent();
      await waitForDataFetch();

      expect(getSelectElement()).toBeInTheDocument();
      consoleErrorSpy.mockRestore();
    });
  });

  describe("Edge cases", () => {
    it("should handle vector stores with all optional fields missing", async () => {
      const minimalStores: VectorStore[] = [
        {
          vector_store_id: "minimal-store",
          custom_llm_provider: "openai",
          created_at: "2024-01-01T00:00:00Z",
          updated_at: "2024-01-01T00:00:00Z",
        },
      ];
      mockVectorStoreListCall.mockResolvedValueOnce({
        data: minimalStores,
      });

      renderComponent();
      await waitForDataFetch();

      expect(screen.getByText("minimal-store (minimal-store)")).toBeInTheDocument();
      expect(screen.getByText("minimal-store (minimal-store)")).not.toHaveAttribute("data-option-description");
    });

    it("should handle very long vector store names", async () => {
      const longNameStores: VectorStore[] = [
        {
          vector_store_id: "store-long",
          custom_llm_provider: "openai",
          vector_store_name: "A".repeat(200),
          created_at: "2024-01-01T00:00:00Z",
          updated_at: "2024-01-01T00:00:00Z",
        },
      ];
      mockVectorStoreListCall.mockResolvedValueOnce({
        data: longNameStores,
      });

      renderComponent();
      await waitForDataFetch();

      expect(screen.getByText(`${"A".repeat(200)} (store-long)`)).toBeInTheDocument();
    });

    it("should handle special characters in vector store names", async () => {
      const specialCharStores: VectorStore[] = [
        {
          vector_store_id: "store-special",
          custom_llm_provider: "openai",
          vector_store_name: 'Store & Co. <Test> "Quotes"',
          created_at: "2024-01-01T00:00:00Z",
          updated_at: "2024-01-01T00:00:00Z",
        },
      ];
      mockVectorStoreListCall.mockResolvedValueOnce({
        data: specialCharStores,
      });

      renderComponent();
      await waitForDataFetch();

      expect(screen.getByText(/Store & Co\. <Test> "Quotes"/)).toBeInTheDocument();
    });
  });
});
