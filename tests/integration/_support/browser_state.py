import json
import sys

from integration._support.database import read_rows

if __name__ == "__main__":
    print(json.dumps(read_rows('SELECT project_id, team_id, models FROM "LiteLLM_VerificationToken" WHERE token=%s', (sys.argv[1],))))
