#!/usr/bin/env python3
"""
Data Validation Web Interface for Vertex AI Supervised Fine-Tuning

A simple web interface to validate training data for fine-tuning jobs.

Usage:
    python data_validation_web_interface.py
    Then open http://localhost:5000 in your browser
"""

import json
import os
import tempfile
from typing import Dict, List, Any
from flask import Flask, render_template_string, request, jsonify, flash
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vertex AI Fine-Tuning Data Validator</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input[type="file"], textarea, select { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
        button { background-color: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        .results { margin-top: 20px; padding: 15px; border-radius: 4px; }
        .success { background-color: #d4edda; border: 1px solid #c3e6cb; color: #155724; }
        .error { background-color: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; }
        .warning { background-color: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }
        .sample-data { background-color: #f8f9fa; padding: 10px; border-radius: 4px; margin: 10px 0; }
        .format-example { margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-left: 4px solid #007bff; }
        pre { background-color: #f8f9fa; padding: 10px; border-radius: 4px; overflow-x: auto; }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }
        .stat-card { background-color: #f8f9fa; padding: 15px; border-radius: 4px; text-align: center; }
        .stat-number { font-size: 24px; font-weight: bold; color: #007bff; }
        .stat-label { color: #666; margin-top: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Vertex AI Fine-Tuning Data Validator</h1>
            <p>Validate your training data for supervised fine-tuning jobs</p>
        </div>

        <form method="POST" enctype="multipart/form-data">
            <div class="form-group">
                <label for="data_format">Data Format:</label>
                <select name="data_format" id="data_format" onchange="showFormatExample()">
                    <option value="jsonl">JSONL (Recommended)</option>
                    <option value="csv">CSV</option>
                    <option value="json">JSON</option>
                </select>
            </div>

            <div class="form-group">
                <label for="file">Upload Training Data File:</label>
                <input type="file" name="file" id="file" accept=".jsonl,.csv,.json,.txt">
            </div>

            <div class="form-group">
                <label for="sample_data">Or paste sample data:</label>
                <textarea name="sample_data" id="sample_data" rows="10" placeholder="Paste your training data here..."></textarea>
            </div>

            <button type="submit">Validate Data</button>
        </form>

        <div id="format-examples" style="margin-top: 20px;">
            <h3>Data Format Examples</h3>
            
            <div id="jsonl-example" class="format-example">
                <h4>JSONL Format (Recommended)</h4>
                <pre>{"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}
{"messages": [{"role": "user", "content": "How do I make coffee?"}, {"role": "assistant", "content": "To make coffee, you need coffee grounds, hot water, and a brewing method..."}]}</pre>
            </div>

            <div id="csv-example" class="format-example" style="display: none;">
                <h4>CSV Format</h4>
                <pre>prompt,completion
"What is the capital of France?","The capital of France is Paris."
"How do I make coffee?","To make coffee, you need coffee grounds, hot water, and a brewing method..."</pre>
            </div>

            <div id="json-example" class="format-example" style="display: none;">
                <h4>JSON Format</h4>
                <pre>[
  {
    "messages": [
      {"role": "user", "content": "What is the capital of France?"},
      {"role": "assistant", "content": "The capital of France is Paris."}
    ]
  },
  {
    "messages": [
      {"role": "user", "content": "How do I make coffee?"},
      {"role": "assistant", "content": "To make coffee, you need coffee grounds, hot water, and a brewing method..."}
    ]
  }
]</pre>
            </div>
        </div>

        {% if results %}
        <div class="results {% if results.is_valid %}success{% else %}error{% endif %}">
            <h3>Validation Results</h3>
            
            {% if results.is_valid %}
            <p>✅ Data is valid for fine-tuning!</p>
            {% else %}
            <p>❌ Data validation failed. Please fix the following issues:</p>
            <ul>
                {% for error in results.errors %}
                <li>{{ error }}</li>
                {% endfor %}
            </ul>
            {% endif %}

            {% if results.warnings %}
            <div class="warning">
                <h4>⚠️ Warnings:</h4>
                <ul>
                    {% for warning in results.warnings %}
                    <li>{{ warning }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}

            {% if results.stats %}
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{{ results.stats.total_examples }}</div>
                    <div class="stat-label">Total Examples</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{{ results.stats.avg_user_length }}</div>
                    <div class="stat-label">Avg User Message Length</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{{ results.stats.avg_assistant_length }}</div>
                    <div class="stat-label">Avg Assistant Message Length</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{{ results.stats.file_size_mb }}</div>
                    <div class="stat-label">File Size (MB)</div>
                </div>
            </div>
            {% endif %}

            {% if results.sample_validated %}
            <div class="sample-data">
                <h4>Sample Validated Data:</h4>
                <pre>{{ results.sample_validated | tojson(indent=2) }}</pre>
            </div>
            {% endif %}
        </div>
        {% endif %}
    </div>

    <script>
        function showFormatExample() {
            const format = document.getElementById('data_format').value;
            
            // Hide all examples
            document.getElementById('jsonl-example').style.display = 'none';
            document.getElementById('csv-example').style.display = 'none';
            document.getElementById('json-example').style.display = 'none';
            
            // Show selected example
            document.getElementById(format + '-example').style.display = 'block';
        }
        
        // Show JSONL example by default
        showFormatExample();
    </script>
</body>
</html>
"""

class DataValidator:
    """Data validator for fine-tuning datasets"""
    
    @staticmethod
    def validate_jsonl_data(data: str) -> Dict[str, Any]:
        """Validate JSONL format data"""
        errors = []
        warnings = []
        examples = []
        
        lines = data.strip().split('\n')
        if not lines:
            errors.append("Empty file")
            return {"is_valid": False, "errors": errors, "warnings": warnings}
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                example = json.loads(line)
                examples.append(example)
                
                # Validate structure
                if "messages" not in example:
                    errors.append(f"Line {i}: Missing 'messages' field")
                    continue
                
                messages = example["messages"]
                if not isinstance(messages, list) or len(messages) < 2:
                    errors.append(f"Line {i}: Messages must be a list with at least 2 messages")
                    continue
                
                # Validate message structure
                for j, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        errors.append(f"Line {i}, message {j}: Message must be an object")
                        continue
                    
                    if "role" not in msg or "content" not in msg:
                        errors.append(f"Line {i}, message {j}: Message must have 'role' and 'content' fields")
                        continue
                    
                    if msg["role"] not in ["user", "assistant", "system"]:
                        errors.append(f"Line {i}, message {j}: Role must be 'user', 'assistant', or 'system'")
                        continue
                    
                    if not isinstance(msg["content"], str):
                        errors.append(f"Line {i}, message {j}: Content must be a string")
                        continue
                
                # Check for user-assistant pattern
                roles = [msg["role"] for msg in messages]
                if not (roles[0] == "user" and roles[-1] == "assistant"):
                    warnings.append(f"Line {i}: Recommended pattern is user message followed by assistant response")
                
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON - {str(e)}")
        
        # Calculate statistics
        stats = DataValidator._calculate_stats(examples)
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "stats": stats,
            "sample_validated": examples[:3] if examples else None
        }
    
    @staticmethod
    def validate_csv_data(data: str) -> Dict[str, Any]:
        """Validate CSV format data"""
        errors = []
        warnings = []
        
        try:
            # Try to parse as CSV
            df = pd.read_csv(pd.StringIO(data))
            
            # Check required columns
            required_columns = ["prompt", "completion"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
                return {"is_valid": False, "errors": errors, "warnings": warnings}
            
            # Check for empty rows
            empty_rows = df[df['prompt'].isna() | df['completion'].isna()]
            if not empty_rows.empty:
                warnings.append(f"Found {len(empty_rows)} rows with empty prompt or completion")
            
            # Convert to messages format for statistics
            examples = []
            for _, row in df.iterrows():
                if pd.notna(row['prompt']) and pd.notna(row['completion']):
                    example = {
                        "messages": [
                            {"role": "user", "content": str(row['prompt'])},
                            {"role": "assistant", "content": str(row['completion'])}
                        ]
                    }
                    examples.append(example)
            
            stats = DataValidator._calculate_stats(examples)
            
            return {
                "is_valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "stats": stats,
                "sample_validated": examples[:3] if examples else None
            }
            
        except Exception as e:
            errors.append(f"Error parsing CSV: {str(e)}")
            return {"is_valid": False, "errors": errors, "warnings": warnings}
    
    @staticmethod
    def validate_json_data(data: str) -> Dict[str, Any]:
        """Validate JSON format data"""
        errors = []
        warnings = []
        
        try:
            examples = json.loads(data)
            
            if not isinstance(examples, list):
                errors.append("Data must be a list of examples")
                return {"is_valid": False, "errors": errors, "warnings": warnings}
            
            # Validate each example
            for i, example in enumerate(examples):
                if not isinstance(example, dict):
                    errors.append(f"Example {i}: Must be an object")
                    continue
                
                if "messages" not in example:
                    errors.append(f"Example {i}: Missing 'messages' field")
                    continue
                
                messages = example["messages"]
                if not isinstance(messages, list) or len(messages) < 2:
                    errors.append(f"Example {i}: Messages must be a list with at least 2 messages")
                    continue
                
                # Validate message structure
                for j, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        errors.append(f"Example {i}, message {j}: Message must be an object")
                        continue
                    
                    if "role" not in msg or "content" not in msg:
                        errors.append(f"Example {i}, message {j}: Message must have 'role' and 'content' fields")
                        continue
                    
                    if msg["role"] not in ["user", "assistant", "system"]:
                        errors.append(f"Example {i}, message {j}: Role must be 'user', 'assistant', or 'system'")
                        continue
                    
                    if not isinstance(msg["content"], str):
                        errors.append(f"Example {i}, message {j}: Content must be a string")
                        continue
            
            stats = DataValidator._calculate_stats(examples)
            
            return {
                "is_valid": len(errors) == 0,
                "errors": errors,
                "warnings": warnings,
                "stats": stats,
                "sample_validated": examples[:3] if examples else None
            }
            
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {str(e)}")
            return {"is_valid": False, "errors": errors, "warnings": warnings}
    
    @staticmethod
    def _calculate_stats(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for the dataset"""
        if not examples:
            return {}
        
        total_examples = len(examples)
        user_lengths = []
        assistant_lengths = []
        
        for example in examples:
            messages = example.get("messages", [])
            for msg in messages:
                if msg.get("role") == "user":
                    user_lengths.append(len(msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    assistant_lengths.append(len(msg.get("content", "")))
        
        return {
            "total_examples": total_examples,
            "avg_user_length": round(sum(user_lengths) / len(user_lengths), 1) if user_lengths else 0,
            "avg_assistant_length": round(sum(assistant_lengths) / len(assistant_lengths), 1) if assistant_lengths else 0,
            "file_size_mb": round(len(str(examples)) / (1024 * 1024), 2)
        }

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    
    if request.method == 'POST':
        data_format = request.form.get('data_format', 'jsonl')
        file = request.files.get('file')
        sample_data = request.form.get('sample_data', '').strip()
        
        # Get data from file or sample
        data = ""
        if file and file.filename:
            data = file.read().decode('utf-8')
        elif sample_data:
            data = sample_data
        else:
            flash('Please provide either a file or sample data', 'error')
            return render_template_string(HTML_TEMPLATE)
        
        # Validate data based on format
        validator = DataValidator()
        if data_format == 'jsonl':
            results = validator.validate_jsonl_data(data)
        elif data_format == 'csv':
            results = validator.validate_csv_data(data)
        elif data_format == 'json':
            results = validator.validate_json_data(data)
    
    return render_template_string(HTML_TEMPLATE, results=results)

if __name__ == '__main__':
    print("🚀 Starting Data Validation Web Interface...")
    print("📱 Open http://localhost:5000 in your browser")
    print("⚠️  This is a development server. Do not use in production.")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 