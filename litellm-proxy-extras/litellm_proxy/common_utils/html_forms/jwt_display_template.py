# JWT display template for SSO debug callback
jwt_display_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LiteLLM SSO Debug - JWT Information</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background-color: #f8fafc;
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            color: #333;
        }

        .container {
            background-color: #fff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            width: 800px;
            max-width: 100%;
        }
        
        .logo-container {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .logo {
            font-size: 24px;
            font-weight: 600;
            color: #1e293b;
        }
        
        h2 {
            margin: 0 0 10px;
            color: #1e293b;
            font-size: 28px;
            font-weight: 600;
            text-align: center;
        }
        
        .subtitle {
            color: #64748b;
            margin: 0 0 20px;
            font-size: 16px;
            text-align: center;
        }

        .info-box {
            background-color: #f1f5f9;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 30px;
            border-left: 4px solid #2563eb;
        }
        
        .success-box {
            background-color: #f0fdf4;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 30px;
            border-left: 4px solid #16a34a;
        }

        .info-header {
            display: flex;
            align-items: center;
            margin-bottom: 12px;
            color: #1e40af;
            font-weight: 600;
            font-size: 16px;
        }
        
        .success-header {
            display: flex;
            align-items: center;
            margin-bottom: 12px;
            color: #166534;
            font-weight: 600;
            font-size: 16px;
        }
        
        .info-header svg, .success-header svg {
            margin-right: 8px;
        }
        
        .data-container {
            margin-top: 20px;
        }
        
        .data-row {
            display: flex;
            border-bottom: 1px solid #e2e8f0;
            padding: 12px 0;
        }
        
        .data-row:last-child {
            border-bottom: none;
        }
        
        .data-label {
            font-weight: 500;
            color: #334155;
            width: 180px;
            flex-shrink: 0;
        }
        
        .data-value {
            color: #475569;
            word-break: break-all;
        }
        
        .jwt-container {
            background-color: #f8fafc;
            border-radius: 6px;
            padding: 15px;
            margin-top: 20px;
            overflow-x: auto;
            border: 1px solid #e2e8f0;
        }
        
        .jwt-text {
            font-family: monospace;
            white-space: pre-wrap;
            word-break: break-all;
            margin: 0;
            color: #334155;
        }
        
        .back-button {
            display: inline-block;
            background-color: #6466E9;
            color: #fff;
            text-decoration: none;
            padding: 10px 16px;
            border-radius: 6px;
            font-weight: 500;
            margin-top: 20px;
            text-align: center;
        }
        
        .back-button:hover {
            background-color: #4138C2;
            text-decoration: none;
        }
        
        .buttons {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        
        .copy-button {
            background-color: #e2e8f0;
            color: #334155;
            border: none;
            padding: 8px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            display: flex;
            align-items: center;
        }
        
        .copy-button:hover {
            background-color: #cbd5e1;
        }
        
        .copy-button svg {
            margin-right: 6px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="logo-container">
            <div class="logo">
                🚅 LiteLLM
            </div>
        </div>
        <h2>SSO Debug Information</h2>
        <p class="subtitle">Results from the SSO authentication process.</p>
        
        <div class="success-box">
            <div class="success-header">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                    <polyline points="22 4 12 14.01 9 11.01"></polyline>
                </svg>
                Authentication Successful
            </div>
            <p>The SSO authentication completed successfully. Below is the information returned by the provider.</p>
        </div>
        
        <div class="data-container" id="userData">
            <!-- Data will be inserted here by JavaScript -->
        </div>
        
        <div class="info-box">
            <div class="info-header">
                <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="10"></circle>
                    <line x1="12" y1="16" x2="12" y2="12"></line>
                    <line x1="12" y1="8" x2="12.01" y2="8"></line>
                </svg>
                JSON Representation
            </div>
            <div class="jwt-container">
                <pre class="jwt-text" id="jsonData">Loading...</pre>
            </div>
            <div class="buttons">
                <button class="copy-button" onclick="copyToClipboard('jsonData')">
                    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                    </svg>
                    Copy to Clipboard
                </button>
            </div>
        </div>
        
        <a href="/sso/debug/login" class="back-button">
            Try Another SSO Login
        </a>
    </div>

    <script>
        // This will be populated with the actual data from the server
        const userData = SSO_DATA;
        
        function renderUserData() {
            const container = document.getElementById('userData');
            const jsonDisplay = document.getElementById('jsonData');
            
            // Format JSON with indentation for display
            jsonDisplay.textContent = JSON.stringify(userData, null, 2);
            
            // Clear container
            container.innerHTML = '';
            
            // Add each key-value pair to the UI
            for (const [key, value] of Object.entries(userData)) {
                if (typeof value !== 'object' || value === null) {
                    const row = document.createElement('div');
                    row.className = 'data-row';
                    
                    const label = document.createElement('div');
                    label.className = 'data-label';
                    label.textContent = key;
                    
                    const dataValue = document.createElement('div');
                    dataValue.className = 'data-value';
                    dataValue.textContent = value !== null ? value : 'null';
                    
                    row.appendChild(label);
                    row.appendChild(dataValue);
                    container.appendChild(row);
                }
            }
        }
        
        function copyToClipboard(elementId) {
            const text = document.getElementById(elementId).textContent;
            navigator.clipboard.writeText(text).then(() => {
                alert('Copied to clipboard!');
            }).catch(err => {
                console.error('Could not copy text: ', err);
            });
        }
        
        // Render the data when the page loads
        document.addEventListener('DOMContentLoaded', renderUserData);
    </script>
</body>
</html>
"""
