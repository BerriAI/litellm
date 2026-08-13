import { act, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import ToolsCard from "./ToolsCard";
import { Tool } from "./types";

describe("ToolsCard", () => {
  const mockTools: Tool[] = [
    {
      name: "Calculator",
      description: "Performs mathematical calculations",
      json: '{"type": "function", "function": {"name": "calculate"}}',
    },
    {
      name: "Weather API",
      description: "Gets current weather information",
      json: '{"type": "function", "function": {"name": "get_weather"}}',
    },
  ];

  const defaultProps = {
    tools: [] as Tool[],
    onAddTool: vi.fn(),
    onEditTool: vi.fn(),
    onRemoveTool: vi.fn(),
  };

  it("should render the component", () => {
    render(<ToolsCard {...defaultProps} />);
    expect(screen.getByText("Tools")).toBeInTheDocument();
  });

  it("should display no tools message when tools array is empty", () => {
    render(<ToolsCard {...defaultProps} />);
    expect(screen.getByText("No tools added")).toBeInTheDocument();
  });

  it("should render tools when provided", () => {
    render(<ToolsCard {...defaultProps} tools={mockTools} />);

    expect(screen.getByText("Calculator")).toBeInTheDocument();
    expect(screen.getByText("Performs mathematical calculations")).toBeInTheDocument();
    expect(screen.getByText("Weather API")).toBeInTheDocument();
    expect(screen.getByText("Gets current weather information")).toBeInTheDocument();
  });

  it("should call onAddTool when Add button is clicked", () => {
    const mockOnAddTool = vi.fn();
    render(<ToolsCard {...defaultProps} onAddTool={mockOnAddTool} />);

    act(() => {
      fireEvent.click(screen.getByRole("button", { name: /add/i }));
    });

    expect(mockOnAddTool).toHaveBeenCalledTimes(1);
  });

  it("should call onEditTool with correct index when Edit button is clicked", () => {
    const mockOnEditTool = vi.fn();
    render(<ToolsCard {...defaultProps} tools={mockTools} onEditTool={mockOnEditTool} />);

    const editButtons = screen.getAllByText("Edit");
    act(() => {
      fireEvent.click(editButtons[0]);
    });

    expect(mockOnEditTool).toHaveBeenCalledWith(0);
    expect(mockOnEditTool).toHaveBeenCalledTimes(1);
  });

  it("should call onRemoveTool with correct index when remove button is clicked", () => {
    const mockOnRemoveTool = vi.fn();
    render(<ToolsCard {...defaultProps} tools={mockTools} onRemoveTool={mockOnRemoveTool} />);

    const removeButtons = screen.getAllByRole("button", { name: "" });
    act(() => {
      fireEvent.click(removeButtons[0]);
    });

    expect(mockOnRemoveTool).toHaveBeenCalledWith(0);
    expect(mockOnRemoveTool).toHaveBeenCalledTimes(1);
  });
});
