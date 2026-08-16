import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import EndpointSelector from "./EndpointSelector";
import { ENDPOINT_OPTIONS } from "./chatConstants";

describe("EndpointSelector", () => {
  Object.values(ENDPOINT_OPTIONS).forEach((endpointType) => {
    it(`should render the endpoint selector for ${endpointType.value}`, async () => {
      const { getByText } = render(<EndpointSelector endpointType={endpointType.value} onEndpointChange={() => {}} />);
      await waitFor(() => {
        expect(getByText(endpointType.label)).toBeInTheDocument();
      });
    });
  });

  it("should filter and show audio endpoints when user inputs 'audio'", async () => {
    const user = userEvent.setup();
    render(<EndpointSelector endpointType={ENDPOINT_OPTIONS[0].value} onEndpointChange={() => {}} />);

    const combobox = screen.getByRole("combobox");
    await user.click(combobox);

    const input = await screen.findByRole("combobox");
    await user.type(input, "audio");

    expect(await screen.findByText("/v1/audio/speech")).toBeInTheDocument();
    expect(await screen.findByText("/v1/audio/transcriptions")).toBeInTheDocument();
  });
});
