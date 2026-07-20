import React from "react";
import { Form, Switch, Tooltip } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { isClientForwardedTokenMode } from "./types";

/**
 * DCR-bridge toggle for the client-forwarded token modes (true_passthrough /
 * oauth_delegate); self-gates to those two auth types and renders nothing
 * otherwise. When on, OAuth-only clients like Claude Desktop can register and
 * sign in through the gateway; when off, the gateway relays the upstream
 * server's own OAuth metadata instead. `initialChecked` seeds the antd
 * Form.Item `initialValue` (not the Switch's DOM defaultChecked): the create
 * form defaults it on, the edit form seeds it from the stored value.
 */
export default function DcrBridgeToggle({
  authType,
  initialChecked,
}: {
  authType?: string | null;
  initialChecked?: boolean;
}) {
  if (!isClientForwardedTokenMode(authType)) return null;
  return (
    <Form.Item
      label={
        <span className="text-sm font-medium text-gray-700 flex items-center">
          Gateway-hosted sign-in (DCR bridge)
          <Tooltip title="Lets OAuth-only clients like Claude Desktop register and sign in through the gateway. Turn off to relay the upstream server's own OAuth metadata instead (for clients pre-registered with the upstream IdP).">
            <InfoCircleOutlined className="ml-2 text-blue-400 hover:text-blue-600 cursor-help" />
          </Tooltip>
        </span>
      }
      name="dcr_bridge"
      valuePropName="checked"
      initialValue={initialChecked}
    >
      <Switch />
    </Form.Item>
  );
}
