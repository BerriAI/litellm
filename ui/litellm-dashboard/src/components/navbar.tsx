"use client";

import Link from 'next/link';
import Image from 'next/image'
import React, { useState } from 'react';
import { useSearchParams } from "next/navigation";
import { Button, Text, Metric,Title, TextInput, Grid, Col } from "@tremor/react";

// Define the props type
interface NavbarProps {
    userID: string | null;
}
const Navbar: React.FC<NavbarProps> = ({ userID }) => {
    const searchParams = useSearchParams();
    const token = searchParams.get("token");
    console.log("User ID:", userID);

    return (
        <nav className="left-0 right-0 top-0 flex justify-between items-center h-12">
            <div className="text-left mx-4 my-2 absolute top-0 left-0">
                <div className="flex flex-col items-center">
                <Link href="/">
                    <button className="text-gray-800 text-2xl px-4 py-1 rounded text-center">🚅 LiteLLM</button>
                </Link>
            </div>
            </div>
            <div className="text-right mx-4 my-2 absolute top-0 right-0">
                <Button color="white">
                <Title>{userID}</Title>
                </Button>
            </div>
            
        </nav>
    )
}

export default Navbar;