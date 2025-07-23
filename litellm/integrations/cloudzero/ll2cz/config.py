# Copyright 2025 CloudZero
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# CHANGELOG: 2025-01-19 - Initial configuration module for ~/.ll2cz/config.yml support (erik.peterson)

"""Configuration management for ll2cz CLI tool."""

from pathlib import Path
from typing import Optional

import yaml
from rich.console import Console
from rich.prompt import Prompt


class Config:
    """Configuration manager for ll2cz settings."""

    def __init__(self):
        """Initialize configuration manager."""
        self.console = Console()
        self.config_dir = Path.home() / ".ll2cz"
        self.config_file = self.config_dir / "config.yml"
        self.config_data = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load configuration from ~/.ll2cz/config.yml if it exists."""
        if not self.config_file.exists():
            return

        try:
            with self.config_file.open('r') as f:
                self.config_data = yaml.safe_load(f) or {}
        except Exception as e:
            self.console.print(f"[yellow]Warning: Failed to load config file {self.config_file}: {e}[/yellow]")
            self.config_data = {}

    def get_database_connection(self, cli_value: Optional[str] = None) -> Optional[str]:
        """Get database connection URL, prioritizing CLI over config."""
        # Treat empty strings as None
        cli_value = cli_value if cli_value and cli_value.strip() else None
        config_value = self.config_data.get('database_url')
        config_value = config_value if config_value and config_value.strip() else None
        return cli_value or config_value

    def get_cz_api_key(self, cli_value: Optional[str] = None) -> Optional[str]:
        """Get CloudZero API key, prioritizing CLI over config."""
        # Treat empty strings as None
        cli_value = cli_value if cli_value and cli_value.strip() else None
        config_value = self.config_data.get('cz_api_key')
        config_value = config_value if config_value and config_value.strip() else None
        return cli_value or config_value

    def get_cz_connection_id(self, cli_value: Optional[str] = None) -> Optional[str]:
        """Get CloudZero connection ID, prioritizing CLI over config."""
        # Treat empty strings as None
        cli_value = cli_value if cli_value and cli_value.strip() else None
        config_value = self.config_data.get('cz_connection_id')
        config_value = config_value if config_value and config_value.strip() else None
        return cli_value or config_value

    def create_example_config(self) -> None:
        """Create an example configuration file."""
        self.config_dir.mkdir(exist_ok=True)

        example_config = {
            'database_url': 'postgresql://user:password@host:5432/litellm_db',
            'cz_api_key': 'your-cloudzero-api-key',
            'cz_connection_id': 'your-connection-id'
        }

        with self.config_file.open('w') as f:
            yaml.dump(example_config, f, default_flow_style=False, sort_keys=False)

        self.console.print(f"[green]Example configuration created at {self.config_file}[/green]")
        self.console.print("[yellow]Please edit the file with your actual values.[/yellow]")

    def show_config_status(self) -> None:
        """Show current configuration status."""
        if not self.config_file.exists():
            self.console.print(f"[yellow]No configuration file found at {self.config_file}[/yellow]")
            return

        self.console.print(f"[blue]Configuration loaded from {self.config_file}[/blue]")

        # Show which values are configured (without revealing secrets)
        configured_items = []
        if self.config_data.get('database_url'):
            configured_items.append("database_url")
        if self.config_data.get('cz_api_key'):
            configured_items.append("cz_api_key")
        if self.config_data.get('cz_connection_id'):
            configured_items.append("cz_connection_id")

        if configured_items:
            self.console.print(f"[green]Configured: {', '.join(configured_items)}[/green]")
        else:
            self.console.print("[yellow]No configuration values found[/yellow]")

    def interactive_edit_config(self) -> None:
        """Interactive configuration editor."""
        self.console.print("\n[bold blue]🔧 Interactive Configuration Editor[/bold blue]")
        self.console.print("[dim]Press Enter to keep current value, or type new value to change it[/dim]\n")

        # Define configuration parameters with descriptions
        config_params = [
            {
                'key': 'database_url',
                'name': 'Database URL',
                'description': 'PostgreSQL connection string for LiteLLM database',
                'example': 'postgresql://user:password@host:5432/litellm_db'
            },
            {
                'key': 'cz_api_key',
                'name': 'CloudZero API Key',
                'description': 'Your CloudZero API key for AnyCost integration',
                'example': 'your-cloudzero-api-key-here'
            },
            {
                'key': 'cz_connection_id',
                'name': 'CloudZero Connection ID',
                'description': 'CloudZero connection ID for data transmission',
                'example': 'your-connection-id-here'
            }
        ]

        # Create backup of current config
        backup_config = self.config_data.copy()
        new_config = self.config_data.copy()
        changes_made = False

        try:
            for param in config_params:
                self._edit_config_parameter(param, new_config)
                if new_config.get(param['key']) != backup_config.get(param['key']):
                    changes_made = True

            # Show summary and save if changes were made
            if changes_made:
                self._show_config_summary(new_config, backup_config)
                if self._confirm_save_changes():
                    self._save_config(new_config)
                    self.console.print("[green]✓ Configuration saved successfully![/green]")
                else:
                    self.console.print("[yellow]Configuration changes discarded.[/yellow]")
            else:
                self.console.print("[dim]No changes made to configuration.[/dim]")

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Configuration editing cancelled.[/yellow]")
        except Exception as e:
            self.console.print(f"[red]Error during configuration editing: {e}[/red]")

    def _edit_config_parameter(self, param: dict, config: dict) -> None:
        """Edit a single configuration parameter."""
        key = param['key']
        name = param['name']
        description = param['description']
        example = param['example']

        # Get current value
        current_value = config.get(key, '')

        # Display parameter info
        self.console.print(f"\n[bold cyan]{name}[/bold cyan]")
        self.console.print(f"[dim]{description}[/dim]")

        if current_value:
            self.console.print(f"Current: [green]{current_value}[/green]")
        else:
            self.console.print("Current: [red]Not set[/red]")
            self.console.print(f"Example: [dim]{example}[/dim]")

        # Get input from user - all input is visible
        prompt_text = f"New {name.lower()}"
        new_value = Prompt.ask(
            prompt_text,
            default=current_value if current_value else "",
            show_default=bool(current_value)
        )

        # Update config if new value provided
        if new_value.strip():
            config[key] = new_value.strip()
        elif not current_value:
            # If no current value and no new value provided, remove key
            config.pop(key, None)

    def _show_config_summary(self, new_config: dict, old_config: dict) -> None:
        """Show a summary of configuration changes."""
        self.console.print("\n[bold yellow]📋 Configuration Changes Summary[/bold yellow]")

        from rich.box import SIMPLE
        from rich.table import Table

        changes_table = Table(show_header=True, header_style="bold cyan", box=SIMPLE, padding=(0, 1))
        changes_table.add_column("Parameter", style="bold blue", no_wrap=False)
        changes_table.add_column("Old Value", style="red", no_wrap=False)
        changes_table.add_column("New Value", style="green", no_wrap=False)
        changes_table.add_column("Status", style="yellow", no_wrap=False)

        param_names = {
            'database_url': 'Database URL',
            'cz_api_key': 'CloudZero API Key',
            'cz_connection_id': 'CloudZero Connection ID'
        }

        for key in ['database_url', 'cz_api_key', 'cz_connection_id']:
            old_val = old_config.get(key, '')
            new_val = new_config.get(key, '')

            if old_val != new_val:
                # Show full values for display
                old_display = old_val if old_val else "[red]Not set[/red]"
                new_display = new_val if new_val else "[red]Removed[/red]"

                if not old_val and new_val:
                    status = "Added"
                elif old_val and not new_val:
                    status = "Removed"
                else:
                    status = "Changed"

                changes_table.add_row(
                    param_names.get(key, key),
                    old_display,
                    new_display,
                    status
                )

        self.console.print(changes_table)


    def _confirm_save_changes(self) -> bool:
        """Confirm if user wants to save the configuration changes."""
        return Prompt.ask(
            "\n[bold]Save these changes?[/bold]",
            choices=["y", "n"],
            default="y"
        ).lower() == "y"

    def _save_config(self, config_data: dict) -> None:
        """Save configuration to file."""
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)

        # Save with nice formatting
        with self.config_file.open('w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False, indent=2)

        # Update in-memory config
        self.config_data = config_data.copy()
