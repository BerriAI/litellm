"use client";

import PtuReservationPanel from "./_components/ptu_reservation_panel";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function PtuReservations() {
  const { accessToken } = useAuthorized();
  return <PtuReservationPanel accessToken={accessToken} />;
}
