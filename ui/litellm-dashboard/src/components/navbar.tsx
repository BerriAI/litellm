"use client";

import Link from "next/link";
import Image from "next/image";
import React, { useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  Button,
  Text,
  Metric,
  Title,
  TextInput,
  Grid,
  Col,
  Card,
} from "@tremor/react";

// Define the props type
interface NavbarProps {
  userID: string | null;
  userRole: string | null;
  userEmail: string | null;
}
const Navbar: React.FC<NavbarProps> = ({ userID, userRole, userEmail }) => {
  console.log("User ID:", userID);
  console.log("userEmail:", userEmail);

  return (
    <nav className="left-0 right-0 top-0 flex justify-between items-center h-12 mb-4">
      <div className="text-left mx-4 my-2 absolute top-0 left-0">
        <div className="flex flex-col items-center">
          <Link href="/">
            <button className="text-gray-800 text-2xl px-4 py-1 rounded text-center">
              🚅 LiteLLM
            </button>
          </Link>
        </div>
      </div>
      <div className="text-right mx-4 my-2 absolute top-0 right-0">
        <Button variant="secondary">
          {userEmail}
          <p>Role: {userRole}</p>
          <p>ID: {userID}</p>
        </Button>
      </div>
    </nav>
  );
};

export default Navbar;
