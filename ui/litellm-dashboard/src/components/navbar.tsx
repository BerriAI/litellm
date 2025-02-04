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
    <>
      <nav className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="w-full px-4">
          <div className="flex justify-between items-center h-14">
            {/* Left side - Just Logo */}
            <div className="flex items-center">
              <Link href="/" className="flex items-center">
                <button className="text-gray-800 rounded text-center">
                  <img
                    src={imageUrl}
                    alt="LiteLLM Brand"
                    className="h-10 w-40 object-contain"
                  />
                </button>
              </Link>
            </div>

            {/* Right side - Links and Admin */}
            <div className="flex items-center space-x-6">
              <a 
                href="https://docs.litellm.ai/docs/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-gray-600 hover:text-gray-800"
              >
                Docs
              </a>
              <Dropdown menu={{ items }}>
                <button className="flex items-center text-sm text-gray-600 hover:text-gray-800">
                  User
                  <svg 
                    className="ml-1 w-4 h-4" 
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
              </Dropdown>
            </div>
          </div>
        </div>
      </nav>
    </>
  );
};

export default Navbar;
