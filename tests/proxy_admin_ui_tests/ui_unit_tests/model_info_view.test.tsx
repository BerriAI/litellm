import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import ModelInfoView from "../../../ui/litellm-dashboard/src/components/model_info_view"

describe("ModelInfoView Component", () => {
  const mockOnModelUpdate = jest.fn()
  const mockOnClose = jest.fn()

  const defaultProps = {
    modelId: "test-model-id",
    onClose: mockOnClose,
    modelData: {
      litellm_params: {
        cache_control_injection_points: [],
      },
      model_info: {},
    },
    accessToken: "test-access-token",
    userID: "test-user-id",
    userRole: "Admin",
    editModel: jest.fn(),
    setEditModalVisible: jest.fn(),
    setSelectedModel: jest.fn(),
    onModelUpdate: mockOnModelUpdate,
    modelAccessGroups: [],
  }

  it("should display 'Disabled' when cache_control_injection_points is an empty array", () => {
    render(<ModelInfoView {...defaultProps} />)

    const cacheControlStatus = screen.getByText("Disabled")
    expect(cacheControlStatus).toBeInTheDocument()
  })

  it("should display 'Enabled' when cache_control_injection_points has values", () => {
    const propsWithCacheControl = {
      ...defaultProps,
      modelData: {
        ...defaultProps.modelData,
        litellm_params: {
          cache_control_injection_points: [{ location: "message", role: "user", index: 0 }],
        },
      },
    }

    render(<ModelInfoView {...propsWithCacheControl} />)

    const cacheControlStatus = screen.getByText("Enabled")
    expect(cacheControlStatus).toBeInTheDocument()
    const cacheControlDetails = screen.getByText(/Location: message/)
    expect(cacheControlDetails).toBeInTheDocument()
  })

  it("should call onModelUpdate with updated cache_control_injection_points as an empty array when disabled", async () => {
    const user = userEvent.setup()
    render(<ModelInfoView {...defaultProps} />)

    const saveButton = screen.getByRole("button", { name: /Save Changes/i })
    await user.click(saveButton)

    expect(mockOnModelUpdate).toHaveBeenCalledWith(
      expect.objectContaining({
        litellm_params: expect.objectContaining({
          cache_control_injection_points: [],
        }),
      }),
    )
  })

  it("should call onModelUpdate with empty cache_control_injection_points when cache control is disabled after being enabled", async () => {
    const user = userEvent.setup()
    const propsWithCacheControlEnabled = {
      ...defaultProps,
      modelData: {
        ...defaultProps.modelData,
        litellm_params: {
          cache_control_injection_points: [{ location: "message", role: "user", index: 0 }],
        },
      },
    }

    render(<ModelInfoView {...propsWithCacheControlEnabled} />)

    const saveButton = screen.getByRole("button", { name: /Save Changes/i })
    await user.click(saveButton)

    expect(mockOnModelUpdate).toHaveBeenCalledWith(
      expect.objectContaining({
        litellm_params: expect.objectContaining({
          cache_control_injection_points: [],
        }),
      }),
    )
  })
})
