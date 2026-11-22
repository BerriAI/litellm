import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import CurlPreview from "./CurlPreview";

const input = {
  baseUrl: "http://localhost:4000",
  endpoint: "/v1/systemone" as const,
  payload: JSON.stringify({ model: "jev-latest", state: "ok" }),
  authHeaderName: "Authorization",
};

describe("CurlPreview", () => {
  it("shows the command and a copy button for a valid payload", () => {
    render(<CurlPreview {...input} />);
    expect(screen.getByLabelText("cURL request")).toHaveTextContent("http://localhost:4000/v1/systemone");
    expect(screen.getByRole("button", { name: /copy curl/i })).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("hides the copy button and points back to the form while the payload has errors", () => {
    render(<CurlPreview {...input} errorCount={2} />);
    expect(screen.getByRole("alert")).toHaveTextContent("Fix 2 issues on the Form tab");
    expect(screen.getByLabelText("cURL request")).toHaveTextContent("jev-latest");
    expect(screen.queryByRole("button", { name: /copy curl/i })).not.toBeInTheDocument();
  });
});
