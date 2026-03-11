import type { RoleMappings as RoleMappingsType } from "@/app/(dashboard)/hooks/sso/useSSOSettings";
import { Card, Divider, Table, Tag, Typography } from "antd";
import { Users } from "lucide-react";
import { defaultRoleDisplayNames } from "./constants";
const { Title, Text } = Typography;

export default function RoleMappings({ roleMappings }: { roleMappings: RoleMappingsType | undefined }) {
  if (!roleMappings) {
    return null;
  }

  const roleMappingsColumns = [
    {
      title: "Role",
      dataIndex: "role",
      key: "role",
      render: (text: string) => <Text strong>{defaultRoleDisplayNames[text]}</Text>,
    },
    {
      title: "Mapped Groups",
      dataIndex: "groups",
      key: "groups",
      render: (groups: string[]) => (
        <>
          {groups.length > 0 ? (
            groups.map((group, index) => (
              <Tag key={index} color="blue">
                {group}
              </Tag>
            ))
          ) : (
            <Text className="text-gray-400 italic">No groups mapped</Text>
          )}
        </>
      ),
    },
  ];
  return (
    <Card>
      <div className="flex items-center gap-3">
        <Users className="w-6 h-6 text-gray-400 mb-2" />
        <Title level={3}>Role Mappings</Title>
      </div>
      <div className="space-y-8">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <Title level={5}>Group Claim</Title>
            <div>
              <Text code>{roleMappings.group_claim}</Text>
            </div>
          </div>
          <div>
            <Title level={5}>Default Role</Title>
            <div>
              <Text strong>{defaultRoleDisplayNames[roleMappings.default_role]}</Text>
            </div>
          </div>
        </div>
        <Divider />
        <Table
          columns={roleMappingsColumns}
          dataSource={Object.entries(roleMappings.roles).map(([role, groups]) => ({
            role,
            groups,
          }))}
          pagination={false}
          bordered
          size="small"
          className="w-full"
        />
      </div>
    </Card>
  );
}
