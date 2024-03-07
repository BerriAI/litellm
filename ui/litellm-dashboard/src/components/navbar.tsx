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
  showSSOBanner: boolean;
}
const Navbar: React.FC<NavbarProps> = ({
  userID,
  userRole,
  userEmail,
  showSSOBanner,
}) => {
  console.log("User ID:", userID);
  console.log("userEmail:", userEmail);

  // const userColors = require('./ui_colors.json') || {};
  const isLocal = process.env.NODE_ENV === "development";
  const imageUrl = isLocal ? "http://localhost:4000/get_image" : "/get_image";

  return (
    <nav className="left-0 right-0 top-0 flex justify-between items-center h-12 mb-4">
      <div className="text-left my-2 absolute top-0 left-0">
        <div className="flex flex-col items-center">
          <Link href="/">
            <button className="text-gray-800 text-2xl py-1 rounded text-center">
              <img
                src={imageUrl}
                width={200}
                height={200}
                alt="LiteLLM Brand"
                className="mr-2"
              />
            </button>
          </Link>
        </div>
      </div>
      <div className="text-right mx-4 my-2 absolute top-0 right-0 flex items-center justify-end space-x-2">
        {showSSOBanner ? (
          <a
            href="https://docs.litellm.ai/docs/proxy/ui#setup-ssoauth-for-ui"
            target="_blank"
            className="mr-2"
          >
            <Button variant="primary" size="lg">
              Enable SSO
            </Button>
          </a>
        ) : null}

        <Button variant="secondary" size="lg">
          {userEmail}
          <p>Role: {userRole}</p>
          <p>ID: {userID}</p>
        </Button>
      </div>
    </nav>
  );
};

export default Navbar;
