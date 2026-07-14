import React, { useState } from "react";
import { Button, Card, Tag, Typography } from "antd";
import TableIconActionButton from "@/components/common_components/IconActionButton/TableIconActionButtons/TableIconActionButton";
import { MoneyCell } from "@/components/shared/table_cells";
import { useIsPtuCostAttributionEnabled } from "@/app/(dashboard)/hooks/ptuReservations/useIsPtuCostAttributionEnabled";
import {
  PtuReservationItem,
  usePtuReservations,
} from "@/app/(dashboard)/hooks/ptuReservations/usePtuReservations";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";
import PtuReservationModal from "./ptu_reservation_modal";
import CloseReservationModal from "./close_reservation_modal";

const { Title, Text } = Typography;

interface PtuReservationPanelProps {
  accessToken: string | null;
}

type ReservationStatus = "active" | "scheduled" | "ended";

function reservationStatus(row: PtuReservationItem, now: Date): ReservationStatus {
  const from = new Date(row.effective_from).getTime();
  const to = row.effective_to ? new Date(row.effective_to).getTime() : null;
  const t = now.getTime();
  if (from > t) return "scheduled";
  if (to !== null && to <= t) return "ended";
  return "active";
}

function statusColor(status: ReservationStatus): string {
  if (status === "active") return "green";
  if (status === "scheduled") return "blue";
  return "default";
}

function monthlyTotal(row: PtuReservationItem): number | null {
  if (row.ptu_count == null || row.cost_per_ptu == null) return null;
  return row.ptu_count * row.cost_per_ptu;
}

const PtuReservationPanel: React.FC<PtuReservationPanelProps> = ({ accessToken }) => {
  const [isCreateVisible, setIsCreateVisible] = useState(false);
  const [isCloseVisible, setIsCloseVisible] = useState(false);
  const [selected, setSelected] = useState<PtuReservationItem | null>(null);

  const { userRole } = useAuthorized();
  const canModify = isProxyAdminRole(userRole ?? "");

  const { enabled: featureEnabled, isLoading: flagLoading } = useIsPtuCostAttributionEnabled();
  const { data: reservations = [], isLoading } = usePtuReservations({}, { enabled: featureEnabled });

  if (flagLoading) {
    return (
      <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
        <Card>
          <Text>Loading&hellip;</Text>
        </Card>
      </div>
    );
  }

  if (!featureEnabled) {
    return (
      <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
        <Card>
          <Title level={4}>PTU Reservations</Title>
          <Text style={{ marginTop: 8, display: "block" }}>
            PTU cost attribution is disabled. Set <code>enable_ptu_cost_attribution: true</code> in{" "}
            <code>general_settings</code> to use this feature.
          </Text>
        </Card>
      </div>
    );
  }

  if (!canModify) {
    return (
      <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
        <Card>
          <Title level={4}>PTU Reservations</Title>
          <Text style={{ marginTop: 8, display: "block" }}>You need proxy-admin access to view PTU reservations.</Text>
        </Card>
      </div>
    );
  }

  const handleCloseClick = (row: PtuReservationItem) => {
    setSelected(row);
    setIsCloseVisible(true);
  };

  const now = new Date();
  const sortedReservations = reservations
    .slice()
    .sort((a, b) => new Date(b.effective_from).getTime() - new Date(a.effective_from).getTime());

  return (
    <div className="w-full mx-auto flex-auto overflow-y-auto m-8 p-2">
      {accessToken && (
        <Button type="primary" size="small" style={{ marginBottom: 8 }} onClick={() => setIsCreateVisible(true)}>
          + Create Reservation
        </Button>
      )}
      <Card>
        <Title level={4}>PTU Reservations</Title>
        <Text>
          Register a flat prepaid cost for a (team, model) window. The daily rollup writes prorated flat cost to each
          day&apos;s team spend row.
        </Text>
        {renderTable(isLoading, sortedReservations, now, handleCloseClick)}
      </Card>

      <PtuReservationModal isModalVisible={isCreateVisible} setIsModalVisible={setIsCreateVisible} />
      <CloseReservationModal
        isModalVisible={isCloseVisible}
        setIsModalVisible={setIsCloseVisible}
        reservation={selected}
      />
    </div>
  );
};

function renderTable(
  isLoading: boolean,
  rows: PtuReservationItem[],
  now: Date,
  handleCloseClick: (r: PtuReservationItem) => void,
) {
  if (isLoading) {
    return <Text style={{ marginTop: 16, display: "block" }}>Loading reservations&hellip;</Text>;
  }
  if (rows.length === 0) {
    return (
      <Text style={{ marginTop: 16, display: "block" }}>
        No reservations yet. Click &quot;Create Reservation&quot; to add one.
      </Text>
    );
  }
  return (
    <table className="w-full mt-4" style={{ borderCollapse: "collapse" }}>
      <thead>
        <tr>
          <th style={cellHead}>Team</th>
          <th style={cellHead}>Model</th>
          <th style={cellHead}>PTU count</th>
          <th style={cellHead}>Cost / PTU</th>
          <th style={cellHead}>Monthly total</th>
          <th style={cellHead}>Effective from</th>
          <th style={cellHead}>Effective to</th>
          <th style={cellHead}>Status</th>
          <th style={cellHead}>Actions</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => {
          const status = reservationStatus(row, now);
          return (
            <tr key={row.id}>
              <td style={cellBody}>{row.team_id}</td>
              <td style={cellBody}>{row.model}</td>
              <td style={cellBody}>{row.ptu_count ?? "—"}</td>
              <td style={cellBody}>
                <MoneyCell value={row.cost_per_ptu} decimals={2} showZero emptyText="—" />
              </td>
              <td style={cellBody}>
                <MoneyCell value={monthlyTotal(row)} decimals={2} showZero emptyText="—" />
              </td>
              <td style={cellBody}>{new Date(row.effective_from).toISOString()}</td>
              <td style={cellBody}>{row.effective_to ? new Date(row.effective_to).toISOString() : "—"}</td>
              <td style={cellBody}>
                <Tag color={statusColor(status)}>{status}</Tag>
              </td>
              <td style={cellBody}>{renderActionCell(status, row, handleCloseClick)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function renderActionCell(
  status: ReservationStatus,
  row: PtuReservationItem,
  handleCloseClick: (r: PtuReservationItem) => void,
) {
  if (status === "ended") {
    return <span>—</span>;
  }
  return (
    <TableIconActionButton
      variant="Delete"
      tooltipText="Close reservation"
      onClick={() => handleCloseClick(row)}
      dataTestId="close-ptu-reservation-button"
    />
  );
}

const cellHead: React.CSSProperties = {
  textAlign: "left",
  padding: "8px",
  borderBottom: "1px solid #e5e7eb",
  fontWeight: 600,
  fontSize: 12,
};

const cellBody: React.CSSProperties = {
  padding: "8px",
  borderBottom: "1px solid #f3f4f6",
  fontSize: 13,
};

export default PtuReservationPanel;
