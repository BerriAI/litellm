"use client";

import Link from "next/link";
import Image from "next/image";
import React, { useEffect, useState } from "react";
import type { MenuProps } from "antd";
import { Dropdown, Space } from "antd";
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
  premiumUser: boolean;
  setProxySettings: React.Dispatch<React.SetStateAction<any>>;
  proxySettings: any;
}
const Navbar: React.FC<NavbarProps> = ({
  userID,
  userRole,
  userEmail,
  premiumUser,
  setProxySettings,
  proxySettings,
}) => {
  console.log("User ID:", userID);
  console.log("userEmail:", userEmail);
  console.log("premiumUser:", premiumUser);

  // const userColors = require('./ui_colors.json') || {};
  const isLocal = process.env.NODE_ENV === "development";
  if (isLocal != true) {
    console.log = function() {};
  }
  const proxyBaseUrl = isLocal ? "http://localhost:4000" : null;
  const imageUrl = isLocal ? "http://localhost:4000/get_image" : "/get_image";
  let logoutUrl = "";

  console.log("PROXY_settings=", proxySettings);

  if (proxySettings) {
    if (proxySettings.PROXY_LOGOUT_URL && proxySettings.PROXY_LOGOUT_URL !== undefined) {
      logoutUrl = proxySettings.PROXY_LOGOUT_URL;
    }
  }

  console.log("logoutUrl=", logoutUrl);

  const handleLogout = () => {
    // Clear cookies
    document.cookie = "token=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;";
    window.location.href = logoutUrl;
  }
   

  const items: MenuProps["items"] = [
    {
      key: "1",
      label: (
        <>
          <p>Role: {userRole}</p>
          <p>ID: {userID}</p>
          <p>Premium User: {String(premiumUser)}</p>
        </>
      ),
    },
    {
      key: "2",
      label: <p onClick={handleLogout}>Logout</p>,
    }
  ];

  return (
    <nav className="left-0 right-0 top-0 flex justify-between items-center h-12 mb-4">
      <div className="text-left my-2 absolute top-0 left-0">
        <div className="flex flex-col items-center">
          <Link href="/">
            <button className="text-gray-800 rounded text-center">
              <img
                src={imageUrl}
                width={160}
                height={160}
                alt="LiteLLM Brand"
                className="mr-2"
              />
            </button>
          </Link>
        </div>
      </div>
      <div className="text-right mx-4 my-2 absolute top-0 right-0 flex items-center justify-end space-x-2">
        {premiumUser ? null : (
          <div
            style={{
              // border: '1px solid #391085',
              padding: "6px",
              borderRadius: "8px", // Added border-radius property
            }}
          >
            <a
              href="https://calendly.com/d/4mp-gd3-k5k/litellm-1-1-onboarding-chat"
              target="_blank"
              style={{
                fontSize: "14px",
                textDecoration: "underline",
              }}
            >
              Get enterprise license
            </a>
          </div>
        )}

        <div
          style={{
            border: "1px solid #391085",
            padding: "6px",
            borderRadius: "8px", // Added border-radius property
          }}
        >
          <Dropdown menu={{ items }}>
            <Space>{userEmail ? userEmail : userRole}</Space>
          </Dropdown>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
