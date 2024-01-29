import Link from 'next/link';
import Image from 'next/image'
import React, { useState } from 'react';
function Navbar() {
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
            <a href="https://docs.litellm.ai/docs/" target="_blank" rel="noopener noreferrer">
                <button className="border border-gray-800 rounded-lg text-gray-800 text-xl px-4 py-1 rounded p-1 mr-2 text-center">Docs</button>
            </a>
            <a href="https://calendly.com/d/4mp-gd3-k5k/berriai-1-1-onboarding-litellm-hosted-version" target="_blank" rel="noopener noreferrer">
                <button className="border border-gray-800 rounded-lg text-gray-800 text-xl px-4 py-1 rounded p-1 text-center">Schedule Demo</button>
            </a>
            </div>
        </nav>
    )
}

export default Navbar;