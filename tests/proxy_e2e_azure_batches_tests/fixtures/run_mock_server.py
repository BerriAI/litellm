
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fixtures.mock_azure_batch_server import create_mock_azure_batch_server
import uvicorn

if __name__ == "__main__":
    app = create_mock_azure_batch_server()
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="info", access_log=False)
