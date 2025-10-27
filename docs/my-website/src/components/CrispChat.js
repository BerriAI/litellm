import React, { useEffect } from 'react';

const CrispChat = () => {
    useEffect(() => {
        window.$crisp = [];
        window.CRISP_WEBSITE_ID = "be07a4d6-dba0-4df7-961d-9302c86b7ebc";

        const d = document;
        const s = d.createElement("script");
        s.src = "https://client.crisp.chat/l.js";
        s.async = 1;
        document.getElementsByTagName("head")[0].appendChild(s);
    }, [])
  
    return null;
};

export default CrispChat;