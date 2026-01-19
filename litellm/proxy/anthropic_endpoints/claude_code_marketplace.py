"""
Claude Code Marketplace Endpoints

Provides a marketplace for Claude Code plugins stored in LiteLLM database.
Enables multi-instance compatible plugin distribution.

Endpoints:
- GET /claude-code/marketplace.json - List all plugins for Claude Code discovery
- GET /claude-code/plugins/{name}/{file} - Serve plugin files
- POST /claude-code/plugins/upload - Upload a plugin (ZIP)
- GET /claude-code/plugins - List plugins (admin)
- POST /claude-code/plugins/{name}/enable - Enable a plugin
- POST /claude-code/plugins/{name}/disable - Disable a plugin
- DELETE /claude-code/plugins/{name} - Delete a plugin
"""

import io
import json
import zipfile
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


async def get_prisma_client():
    """Get the prisma client from proxy_server"""
    from litellm.proxy.proxy_server import prisma_client
    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail="Database not connected. Please set DATABASE_URL."
        )
    return prisma_client


@router.get(
    "/claude-code/marketplace.json",
    tags=["Claude Code Marketplace"],
)
async def get_marketplace():
    """
    Serve marketplace.json for Claude Code plugin discovery.

    This endpoint is accessed by Claude Code CLI when users run:
    - claude marketplace add <name> <url>
    - claude plugin install <name>@<marketplace>

    Example:
        claude marketplace add litellm http://localhost:4000
        claude plugin install hello-world@litellm
    """
    try:
        prisma_client = await get_prisma_client()

        # Get all enabled plugins from database
        plugins = await prisma_client.db.litellm_claudecodeplugintable.find_many(
            where={"enabled": True}
        )

        marketplace = []
        for plugin in plugins:
            # Parse manifest
            try:
                manifest = json.loads(plugin.manifest_json)
            except json.JSONDecodeError:
                continue

            entry = {
                "name": plugin.name,
                "source": f"./plugins/{plugin.name}",
            }

            # Add fields from manifest/plugin
            if plugin.version:
                entry["version"] = plugin.version
            if plugin.description:
                entry["description"] = plugin.description
            if "author" in manifest:
                entry["author"] = manifest["author"]
            if "homepage" in manifest:
                entry["homepage"] = manifest["homepage"]
            if "keywords" in manifest:
                entry["keywords"] = manifest["keywords"]

            marketplace.append(entry)

        return JSONResponse(content=marketplace)

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error generating marketplace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/claude-code/plugins/{plugin_name}/{file_path:path}",
    tags=["Claude Code Marketplace"],
)
async def get_plugin_file(plugin_name: str, file_path: str):
    """
    Serve plugin files from database.

    Claude Code will request files like:
    - /claude-code/plugins/my-plugin/.claude-plugin/plugin.json
    - /claude-code/plugins/my-plugin/commands/hello.md
    """
    try:
        prisma_client = await get_prisma_client()

        # Get plugin from database
        plugin = await prisma_client.db.litellm_claudecodeplugintable.find_unique(
            where={"name": plugin_name}
        )

        if not plugin or not plugin.enabled:
            raise HTTPException(status_code=404, detail="Plugin not found")

        # Parse files JSON
        try:
            files = json.loads(plugin.files_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Invalid plugin files data")

        # Get requested file
        if file_path not in files:
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

        content = files[file_path]

        # Determine media type
        media_type = "text/plain"
        if file_path.endswith(".json"):
            media_type = "application/json"
        elif file_path.endswith(".md"):
            media_type = "text/markdown"
        elif file_path.endswith(".sh"):
            media_type = "application/x-sh"
        elif file_path.endswith(".py"):
            media_type = "text/x-python"
        elif file_path.endswith(".js"):
            media_type = "text/javascript"

        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f'inline; filename="{file_path.split("/")[-1]}"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error serving plugin file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/claude-code/plugins/upload",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
)
async def upload_plugin(
    file: UploadFile = File(...),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Upload a Claude Code plugin as a ZIP file.

    The ZIP should contain:
    - .claude-plugin/plugin.json (required)
    - commands/*.md (optional)
    - agents/*.md (optional)
    - skills/*/SKILL.md (optional)
    - hooks/hooks.json (optional)

    Example:
        curl -X POST \\
          -H "Authorization: Bearer sk-..." \\
          -F "file=@my-plugin.zip" \\
          http://localhost:4000/claude-code/plugins/upload
    """
    if not file.filename or not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")

    try:
        prisma_client = await get_prisma_client()

        # Read ZIP
        content = await file.read()
        zip_buffer = io.BytesIO(content)

        files = {}
        with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if not file_info.is_dir():
                    file_path = file_info.filename
                    # Security: block path traversal
                    if ".." in file_path or file_path.startswith("/"):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid path in ZIP: {file_path}"
                        )
                    file_content = zip_ref.read(file_info).decode('utf-8', errors='ignore')
                    files[file_path] = file_content

        # Validate manifest exists
        manifest_path = ".claude-plugin/plugin.json"
        if manifest_path not in files:
            raise HTTPException(
                status_code=400,
                detail="ZIP must contain .claude-plugin/plugin.json"
            )

        # Parse manifest
        try:
            manifest = json.loads(files[manifest_path])
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid plugin.json: {e}"
            )

        # Validate required fields
        if "name" not in manifest:
            raise HTTPException(
                status_code=400,
                detail="plugin.json must contain 'name' field"
            )

        plugin_name = manifest["name"]
        plugin_version = manifest.get("version", "1.0.0")
        plugin_description = manifest.get("description")

        # Validate name format (kebab-case)
        import re
        if not re.match(r"^[a-z0-9-]+$", plugin_name):
            raise HTTPException(
                status_code=400,
                detail="Plugin name must be kebab-case (lowercase letters, numbers, hyphens)"
            )

        # Check if plugin exists
        existing = await prisma_client.db.litellm_claudecodeplugintable.find_unique(
            where={"name": plugin_name}
        )

        if existing:
            # Update existing plugin
            plugin = await prisma_client.db.litellm_claudecodeplugintable.update(
                where={"name": plugin_name},
                data={
                    "version": plugin_version,
                    "description": plugin_description,
                    "manifest_json": json.dumps(manifest),
                    "files_json": json.dumps(files),
                    "updated_at": datetime.now(timezone.utc)
                }
            )
            action = "updated"
        else:
            # Create new plugin
            plugin = await prisma_client.db.litellm_claudecodeplugintable.create(
                data={
                    "name": plugin_name,
                    "version": plugin_version,
                    "description": plugin_description,
                    "manifest_json": json.dumps(manifest),
                    "files_json": json.dumps(files),
                    "enabled": True,
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                    "created_by": user_api_key_dict.user_id
                }
            )
            action = "created"

        verbose_proxy_logger.info(f"Plugin {plugin_name} {action} successfully")

        return {
            "status": "success",
            "action": action,
            "plugin_id": plugin.id,
            "plugin_name": plugin.name,
            "version": plugin.version,
            "files_count": len(files)
        }

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error uploading plugin: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get(
    "/claude-code/plugins",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_plugins(
    enabled_only: bool = False,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """List all plugins in the marketplace"""
    try:
        prisma_client = await get_prisma_client()

        where = {"enabled": True} if enabled_only else {}
        plugins = await prisma_client.db.litellm_claudecodeplugintable.find_many(
            where=where,
            order_by={"created_at": "desc"}
        )

        return {
            "plugins": [
                {
                    "id": p.id,
                    "name": p.name,
                    "version": p.version,
                    "description": p.description,
                    "enabled": p.enabled,
                    "created_at": p.created_at.isoformat() if p.created_at else None,
                    "updated_at": p.updated_at.isoformat() if p.updated_at else None,
                }
                for p in plugins
            ],
            "count": len(plugins)
        }
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error listing plugins: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/claude-code/plugins/{plugin_name}/enable",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
)
async def enable_plugin(
    plugin_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Enable a disabled plugin"""
    try:
        prisma_client = await get_prisma_client()

        plugin = await prisma_client.db.litellm_claudecodeplugintable.find_unique(
            where={"name": plugin_name}
        )
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        await prisma_client.db.litellm_claudecodeplugintable.update(
            where={"name": plugin_name},
            data={"enabled": True, "updated_at": datetime.now(timezone.utc)}
        )

        verbose_proxy_logger.info(f"Plugin {plugin_name} enabled")
        return {"status": "success", "message": f"Plugin {plugin_name} enabled"}
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error enabling plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/claude-code/plugins/{plugin_name}/disable",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
)
async def disable_plugin(
    plugin_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Disable a plugin without deleting it"""
    try:
        prisma_client = await get_prisma_client()

        plugin = await prisma_client.db.litellm_claudecodeplugintable.find_unique(
            where={"name": plugin_name}
        )
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        await prisma_client.db.litellm_claudecodeplugintable.update(
            where={"name": plugin_name},
            data={"enabled": False, "updated_at": datetime.now(timezone.utc)}
        )

        verbose_proxy_logger.info(f"Plugin {plugin_name} disabled")
        return {"status": "success", "message": f"Plugin {plugin_name} disabled"}
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error disabling plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/claude-code/plugins/{plugin_name}",
    tags=["Claude Code Marketplace"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_plugin(
    plugin_name: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Delete a plugin from the marketplace"""
    try:
        prisma_client = await get_prisma_client()

        plugin = await prisma_client.db.litellm_claudecodeplugintable.find_unique(
            where={"name": plugin_name}
        )
        if not plugin:
            raise HTTPException(status_code=404, detail="Plugin not found")

        await prisma_client.db.litellm_claudecodeplugintable.delete(
            where={"name": plugin_name}
        )

        verbose_proxy_logger.info(f"Plugin {plugin_name} deleted")
        return {"status": "success", "message": f"Plugin {plugin_name} deleted"}
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"Error deleting plugin: {e}")
        raise HTTPException(status_code=500, detail=str(e))
