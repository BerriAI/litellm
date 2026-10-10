import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";
import WhatsNewBanner, { WHATS_NEW_ITEMS } from "./WhatsNewBanner";

const STORAGE_KEY = "hideWhatsNewBanner";

describe("WhatsNewBanner", () => {
  beforeEach(() => {
    localStorage.removeItem(STORAGE_KEY);
  });

  it("renders one external link per item pointing at its post", () => {
    render(<WhatsNewBanner />);
    for (const item of WHATS_NEW_ITEMS) {
      const link = screen.getByRole("link", { name: new RegExp(`^${item.title}`) });
      expect(link).toHaveAttribute("href", item.href);
      expect(link).toHaveAttribute("target", "_blank");
      expect(link).toHaveAttribute("rel", "noopener noreferrer");
    }
    expect(screen.getAllByRole("link")).toHaveLength(WHATS_NEW_ITEMS.length);
  });

  it("hides itself and persists the dismissal when the dismiss button is clicked", () => {
    render(<WhatsNewBanner />);
    expect(screen.getByText("What's new")).toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Dismiss what's new"));

    expect(screen.queryByText("What's new")).not.toBeInTheDocument();
    expect(localStorage.getItem(STORAGE_KEY)).toBe("true");
  });

  it("stays dismissed on remount once persisted", () => {
    localStorage.setItem(STORAGE_KEY, "true");
    render(<WhatsNewBanner />);
    expect(screen.queryByText("What's new")).not.toBeInTheDocument();
  });
});
