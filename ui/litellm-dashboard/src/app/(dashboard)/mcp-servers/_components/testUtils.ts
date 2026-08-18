import { act, fireEvent, screen, waitFor } from "@testing-library/react";
import { expect } from "vitest";

export async function selectAntOption(labelText: string, optionText: string) {
  const label = screen.getByText(labelText);
  const select =
    label.closest(".ant-form-item")?.querySelector(".ant-select") ??
    label.closest(".ant-collapse-item")?.querySelector(".ant-select") ??
    label.closest("div")?.querySelector(".ant-select") ??
    null;

  act(() => {
    fireEvent.mouseDown(select!.querySelector(".ant-select-selector")!);
  });

  await waitFor(() => {
    expect(document.querySelectorAll(".ant-select-item-option").length).toBeGreaterThan(0);
  });

  const option = Array.from(document.querySelectorAll(".ant-select-item-option")).find((el) =>
    el.textContent?.includes(optionText),
  );
  expect(option).toBeTruthy();
  act(() => {
    fireEvent.click(option!);
  });
}
