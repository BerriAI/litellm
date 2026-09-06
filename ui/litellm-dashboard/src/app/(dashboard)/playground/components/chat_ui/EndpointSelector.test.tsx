import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import EndpointSelector from "./EndpointSelector";
import { ENDPOINT_OPTIONS } from "./chatConstants";

describe("EndpointSelector", () => {
  Object.values(ENDPOINT_OPTIONS).forEach((endpointType) => {
    it(`should render the endpoint selector for ${endpointType.value}`, async () => {
      render(<EndpointSelector endpointType={endpointType.value} onEndpointChange={() => {}} />);
      await waitFor(() => {
        expect(screen.getByRole("combobox")).toHaveValue(endpointType.label);
      });
    });
  });

  it("should filter and show audio endpoints when user inputs 'audio'", async () => {
    const user = userEvent.setup();
    render(<EndpointSelector endpointType={ENDPOINT_OPTIONS[0].value} onEndpointChange={() => {}} />);

    const input = screen.getByRole("combobox");
    await user.click(input);
    await user.clear(input);
    fireEvent.change(input, { target: { value: "audio" } });

    expect(await screen.findByText("/v1/audio/speech")).toBeInTheDocument();
    expect(await screen.findByText("/v1/audio/transcriptions")).toBeInTheDocument();
  });
});
