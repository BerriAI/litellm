"use client";
import React, { Suspense, useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";
import Navbar from "../../components/navbar";
import Sidebar from "../../components/leftnav";
import { jwtDecode } from "jwt-decode";
import Link from "next/link";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeaderCell,
  TableRow,
  Card,
  Icon,
  Button,
  Col,
  Grid,
} from "@tremor/react";
import { CogIcon } from "@heroicons/react/outline";

const TeamSettingsPage = () => {
  const [userRole, setUserRole] = useState("");
  const [userEmail, setUserEmail] = useState<null | string>(null);
  const [teams, setTeams] = useState<null | string[]>(null);
  const searchParams = useSearchParams();

  const userID = searchParams.get("userID");
  const token = searchParams.get("token");

  const [page, setPage] = useState("team");
  const [accessToken, setAccessToken] = useState<string | null>(null);

  useEffect(() => {
    if (token) {
      const decoded = jwtDecode(token) as { [key: string]: any };
      if (decoded) {
        // cast decoded to dictionary
        console.log("Decoded token:", decoded);

        console.log("Decoded key:", decoded.key);
        // set accessToken
        setAccessToken(decoded.key);

        // check if userRole is defined
        if (decoded.user_role) {
          const formattedUserRole = formatUserRole(decoded.user_role);
          console.log("Decoded user_role:", formattedUserRole);
          setUserRole(formattedUserRole);
        } else {
          console.log("User role not defined");
        }

        if (decoded.user_email) {
          setUserEmail(decoded.user_email);
        } else {
          console.log(`User Email is not set ${decoded}`);
        }
      }
    }
  }, [token]);

  function formatUserRole(userRole: string) {
    if (!userRole) {
      return "Undefined Role";
    }
    console.log(`Received user role: ${userRole}`);
    switch (userRole.toLowerCase()) {
      case "app_owner":
        return "App Owner";
      case "demo_app_owner":
        return "App Owner";
      case "app_admin":
        return "Admin";
      case "app_user":
        return "App User";
      default:
        return "Unknown Role";
    }
  }

  return (
    <Suspense fallback={<div>Loading...</div>}>
      <div className="flex flex-col min-h-screen">
        <Navbar userID={userID} userRole={userRole} userEmail={userEmail} />
        <div className="flex flex-1 overflow-auto">
          <Grid numItems={1} className="gap-0 p-10 h-[75vh] w-full">
            <Col numColSpan={1}>
              <Card className="w-full mx-auto flex-auto overflow-y-auto max-h-[50vh]">
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableHeaderCell>Team Name</TableHeaderCell>
                      <TableHeaderCell>Spend (USD)</TableHeaderCell>
                      <TableHeaderCell>Budget (USD)</TableHeaderCell>
                      <TableHeaderCell>TPM / RPM Limits</TableHeaderCell>
                      <TableHeaderCell>Settings</TableHeaderCell>
                    </TableRow>
                  </TableHead>

                  <TableBody>
                    <TableRow>
                      <TableCell>Wilhelm Tell</TableCell>
                      <TableCell className="text-right">1</TableCell>
                      <TableCell>Uri, Schwyz, Unterwalden</TableCell>
                      <TableCell>National Hero</TableCell>
                      <TableCell>
                        <Icon icon={CogIcon} size="sm" />
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>The Witcher</TableCell>
                      <TableCell className="text-right">129</TableCell>
                      <TableCell>Kaedwen</TableCell>
                      <TableCell>Legend</TableCell>
                      <TableCell>
                        <Icon icon={CogIcon} size="sm" />
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell>Mizutsune</TableCell>
                      <TableCell className="text-right">82</TableCell>
                      <TableCell>Japan</TableCell>
                      <TableCell>N/A</TableCell>
                      <TableCell>
                        <Icon icon={CogIcon} size="sm" />
                      </TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </Card>
            </Col>
            <Col numColSpan={1}>
              <Link
                href={`/team?userID=${searchParams.get(
                  "userID"
                )}&token=${searchParams.get("token")}`}
              >
                <Button className="mx-auto">+ Create New Team</Button>
              </Link>
            </Col>
          </Grid>
        </div>
      </div>
    </Suspense>
  );
};
export default TeamSettingsPage;
