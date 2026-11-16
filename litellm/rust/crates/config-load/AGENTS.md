# Config load

- This crate reads config files: the root file and every file its `include` names, merged breadth first, each file once. The typed `Config` lives in `litellm-config`, which reads no file and gets the merged document through `Config::from_value`
- A data crate never depends on this one. Only a host that starts from a path on disk (the gateway) does
