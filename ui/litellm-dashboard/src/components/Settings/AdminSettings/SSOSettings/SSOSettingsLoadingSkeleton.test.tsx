import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import SSOSettingsLoadingSkeleton from "./SSOSettingsLoadingSkeleton";

// Mock lucide-react icons
vi.mock("lucide-react", () => ({
  Shield: ({ className }: any) => <div data-testid="shield-icon" className={className} />,
}));

// Mock Ant Design components
vi.mock("antd", () => ({
  Card: ({ children, ...props }: any) => (
    <div data-testid="card" {...props}>
      {children}
    </div>
  ),
  Descriptions: Object.assign(
    ({ children, bordered, column, ...props }: any) => (
      <div data-testid="descriptions" data-bordered={bordered} data-column={JSON.stringify(column)} {...props}>
        {children}
      </div>
    ),
    {
      Item: ({ children, label, ...props }: any) => (
        <div data-testid="descriptions-item" {...props}>
          <div data-testid="descriptions-item-label">{label}</div>
          <div data-testid="descriptions-item-content">{children}</div>
        </div>
      ),
    },
  ),
  Typography: {
    Title: ({ children, level, ...props }: any) => (
      <div data-testid="typography-title" data-level={level} {...props}>
        {children}
      </div>
    ),
    Text: ({ children, type, ...props }: any) => (
      <div data-testid="typography-text" data-type={type} {...props}>
        {children}
      </div>
    ),
  },
  Space: ({ children, direction, size, className, ...props }: any) => (
    <div data-testid="space" data-direction={direction} data-size={size} className={className} {...props}>
      {children}
    </div>
  ),
  Skeleton: {
    Button: ({ active, size, style, ...props }: any) => (
      <div
        data-testid="skeleton-button"
        data-active={active}
        data-size={size}
        data-style={JSON.stringify(style)}
        {...props}
      >
        Button Skeleton
      </div>
    ),
    Node: ({ active, style, ...props }: any) => (
      <div data-testid="skeleton-node" data-active={active} data-style={JSON.stringify(style)} {...props}>
        Node Skeleton
      </div>
    ),
  },
}));

describe("SSOSettingsLoadingSkeleton", () => {
  it("should render without crashing", () => {
    expect(() => render(<SSOSettingsLoadingSkeleton />)).not.toThrow();
  });

  it("should render Card component", () => {
    render(<SSOSettingsLoadingSkeleton />);
    expect(screen.getByTestId("card")).toBeInTheDocument();
  });

  it("should render Space component with correct props", () => {
    render(<SSOSettingsLoadingSkeleton />);
    const space = screen.getByTestId("space");
    expect(space).toBeInTheDocument();
    expect(space).toHaveAttribute("data-direction", "vertical");
    expect(space).toHaveAttribute("data-size", "large");
    expect(space).toHaveClass("w-full");
  });

  describe("Header Section", () => {
    it("should render Shield icon", () => {
      render(<SSOSettingsLoadingSkeleton />);
      const shieldIcon = screen.getByTestId("shield-icon");
      expect(shieldIcon).toBeInTheDocument();
      expect(shieldIcon).toHaveClass("w-6 h-6 text-gray-400");
    });

    it("should render title with correct text and level", () => {
      render(<SSOSettingsLoadingSkeleton />);
      const title = screen.getByTestId("typography-title");
      expect(title).toBeInTheDocument();
      expect(title).toHaveAttribute("data-level", "3");
      expect(title).toHaveTextContent("SSO Configuration");
    });

    it("should render subtitle text", () => {
      render(<SSOSettingsLoadingSkeleton />);
      const text = screen.getByTestId("typography-text");
      expect(text).toBeInTheDocument();
      expect(text).toHaveAttribute("data-type", "secondary");
      expect(text).toHaveTextContent("Manage Single Sign-On authentication settings");
    });

    it("should render two skeleton buttons with correct styles", () => {
      render(<SSOSettingsLoadingSkeleton />);
      const buttons = screen.getAllByTestId("skeleton-button");
      expect(buttons).toHaveLength(2);

      // First button
      expect(buttons[0]).toHaveAttribute("data-active", "true");
      expect(buttons[0]).toHaveAttribute("data-size", "default");
      expect(buttons[0]).toHaveAttribute("data-style", JSON.stringify({ width: 170, height: 32 }));

      // Second button
      expect(buttons[1]).toHaveAttribute("data-active", "true");
      expect(buttons[1]).toHaveAttribute("data-size", "default");
      expect(buttons[1]).toHaveAttribute("data-style", JSON.stringify({ width: 190, height: 32 }));
    });
  });

  describe("Descriptions Table", () => {
    it("should render Descriptions component with bordered prop", () => {
      render(<SSOSettingsLoadingSkeleton />);
      const descriptions = screen.getByTestId("descriptions");
      expect(descriptions).toBeInTheDocument();
      expect(descriptions).toHaveAttribute("data-bordered", "true");
    });

    it("should apply correct column configuration", () => {
      render(<SSOSettingsLoadingSkeleton />);
      const descriptions = screen.getByTestId("descriptions");
      const expectedColumn = {
        xxl: 1,
        xl: 1,
        lg: 1,
        md: 1,
        sm: 1,
        xs: 1,
      };
      expect(descriptions).toHaveAttribute("data-column", JSON.stringify(expectedColumn));
    });

    it("should render exactly 5 description items", () => {
      render(<SSOSettingsLoadingSkeleton />);
      const items = screen.getAllByTestId("descriptions-item");
      expect(items).toHaveLength(5);
    });

    describe("Description Items Structure", () => {
      it("should render exactly 10 skeleton nodes total", () => {
        render(<SSOSettingsLoadingSkeleton />);
        const skeletonNodes = screen.getAllByTestId("skeleton-node");
        expect(skeletonNodes).toHaveLength(10);
      });

      it("should render 5 skeleton nodes for labels with width 80", () => {
        render(<SSOSettingsLoadingSkeleton />);
        const skeletonNodes = screen.getAllByTestId("skeleton-node");

        const labelNodes = skeletonNodes.filter(
          (node) => node.getAttribute("data-style") === JSON.stringify({ width: 80, height: 16 }),
        );
        expect(labelNodes).toHaveLength(5);

        labelNodes.forEach((node) => {
          expect(node).toHaveAttribute("data-active", "true");
        });
      });

      it("should render skeleton nodes for content with correct widths", () => {
        render(<SSOSettingsLoadingSkeleton />);
        const skeletonNodes = screen.getAllByTestId("skeleton-node");

        // Expected content widths: [100, 200, 250, 180, 220]
        const expectedWidths = [100, 200, 250, 180, 220];
        expectedWidths.forEach((width) => {
          const contentNode = skeletonNodes.find(
            (node) => node.getAttribute("data-style") === JSON.stringify({ width, height: 16 }),
          );
          expect(contentNode).toBeInTheDocument();
          expect(contentNode).toHaveAttribute("data-active", "true");
        });
      });
    });
  });

  describe("Accessibility and Structure", () => {
    it("should have proper semantic structure", () => {
      render(<SSOSettingsLoadingSkeleton />);
      // Card contains Space
      const card = screen.getByTestId("card");
      const space = screen.getByTestId("space");
      expect(card).toContainElement(space);

      // Space contains header section and descriptions
      const descriptions = screen.getByTestId("descriptions");
      expect(space).toContainElement(descriptions);
    });

    it("should render all skeleton elements as active", () => {
      render(<SSOSettingsLoadingSkeleton />);
      const skeletonNodes = screen.getAllByTestId("skeleton-node");
      const skeletonButtons = screen.getAllByTestId("skeleton-button");

      skeletonNodes.forEach((node) => {
        expect(node).toHaveAttribute("data-active", "true");
      });

      skeletonButtons.forEach((button) => {
        expect(button).toHaveAttribute("data-active", "true");
      });
    });
  });
});
