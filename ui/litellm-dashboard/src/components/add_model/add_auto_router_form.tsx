import React from "react"
import { Card, Form, message } from "antd"
import AddAutoRouterTab from "./add_auto_router_tab"
import { handleAddAutoRouterSubmit } from "./handle_add_auto_router_submit"

interface AddAutoRouterFormProps {
  onSuccess: () => void
  onError: (error: string) => void
  accessToken: string
  userRole: string
}

const AddAutoRouterForm: React.FC<AddAutoRouterFormProps> = ({ onSuccess, onError, accessToken, userRole }) => {
  // Create separate form instance for auto router
  const [form] = Form.useForm()

  // Handle auto router submission
  const handleSubmit = () => {
    form
      .validateFields()
      .then(async (values) => {
        try {
          console.log("=== AUTO ROUTER FORM SUBMISSION ===")
          console.log("Form values:", values)

          await handleAddAutoRouterSubmit(values, accessToken, form, onSuccess)
          message.success("Auto Router added successfully!")
          form.resetFields()
        } catch (error) {
          console.error("Add Auto Router submission failed:", error)
          const errorMessage = error instanceof Error ? error.message : "Failed to add auto router"
          message.error(errorMessage)
          onError(errorMessage)
        }
      })
      .catch((error) => {
        console.error("Validation failed:", error)
      })
  }

  return (
    <Card>
      <AddAutoRouterTab form={form} handleOk={handleSubmit} accessToken={accessToken} userRole={userRole} />
    </Card>
  )
}

export default AddAutoRouterForm
