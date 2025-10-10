import React, { FC } from "react";

import { Organization, EditModalProps, OrganizationMember } from "./types";
import { Card, Col, Table, TableHead, TableHeaderCell, TableBody, TableRow, TableCell } from "@tremor/react";

interface Member {
  user_email?: string;
  user_id?: string;
  role: string;
}

interface MemberListTableProps {
  selectedEntity?: Organization;
  onEditSubmit: (entity: Organization) => void;
  editModalComponent: React.ComponentType<EditModalProps>;
  entityType: "team" | "organization";
}

const MemberListTable: FC<MemberListTableProps> = ({
  selectedEntity,
  onEditSubmit,
  editModalComponent: EditModal,
  entityType,
}) => {
  const [editModalVisible, setEditModalVisible] = React.useState(false);

  const handleEditCancel = () => {
    setEditModalVisible(false);
  };

  const handleEditSubmit = (entity: Organization) => {
    onEditSubmit(entity);
    setEditModalVisible(false);
  };

  const getMemberIdentifier = (member: Member) => {
    return member.user_email || member.user_id || "Unknown Member";
  };

  return (
    <Col numColSpan={1}>
      <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh]">
        <Table>
          <TableHead>
            <TableRow>
              <TableHeaderCell>{entityType === "team" ? "Team Member" : "Organization Member"}</TableHeaderCell>
              <TableHeaderCell>Role</TableHeaderCell>
            </TableRow>
          </TableHead>

          <TableBody>
            {(selectedEntity?.members ?? []).map((value: OrganizationMember, index: number) => (
              <TableRow key={`${value.user_id}-${index}`}>
                <TableCell>{value.user_id}</TableCell>
                <TableCell>{value.user_role}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>

      {selectedEntity && (
        <EditModal
          visible={editModalVisible}
          onCancel={handleEditCancel}
          entity={selectedEntity}
          onSubmit={handleEditSubmit}
        />
      )}
    </Col>
  );
};

export default MemberListTable;
