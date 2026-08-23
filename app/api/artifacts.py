"""Artifact routes: list/create/delete a session's artifacts, the accept/refine
workflow (approve an artifact or edit its content), and exporting the final,
approved set as a downloadable Markdown report.

An artifact always starts 'pending' (whether saved manually from a chat
message or auto-extracted from an uploaded PDF) and only counts as project
context for the RAG engine, and appears in the export, once 'approved' - see
ChatDatabase.get_session_artifacts and RagEngine.generate_response.
"""
from fastapi import APIRouter, Form
from fastapi.responses import PlainTextResponse

from app.core.database import db

router = APIRouter()

# Maps the internal artifact type codes to the section headings used in the export.
TYPE_HEADINGS = {
    "Asset": "Assets",
    "Threat": "Threats",
    "Goal": "Security Goals",
    "Req": "Security Requirements",
}


@router.get("/sessions/{session_id}/artifacts")
async def list_artifacts(session_id: str):
    artifacts = db.get_session_artifacts(session_id)
    return {"artifacts": artifacts}


@router.delete("/artifacts/{artifact_id}")
async def remove_artifact(artifact_id: int):
    """Removes a specific artifact (used for both 'Discard' on a pending
    artifact and 'Remove' on an approved one)."""
    db.delete_single_artifact(artifact_id)
    return {"status": "removed"}


@router.post("/artifacts")
async def create_artifact(
    session_id: str = Form(...),
    artifact_type: str = Form(...),
    content: str = Form(...)
):
    """Saves an artifact as 'pending' - e.g. from the "Save as {Step}" button
    under a chat message. Must be explicitly approved in the dashboard before
    it becomes project context."""
    db.save_artifact(session_id, artifact_type, content)
    return {"status": "saved"}


@router.post("/artifacts/{artifact_id}/approve")
async def approve_artifact_route(artifact_id: int):
    """Marks an artifact as approved/final, making it count as project context."""
    db.approve_artifact(artifact_id)
    return {"status": "approved"}


@router.post("/artifacts/{artifact_id}/edit")
async def edit_artifact_route(artifact_id: int, content: str = Form(...)):
    """Refines the content of an existing artifact without changing its status."""
    db.update_artifact_content(artifact_id, content)
    return {"status": "updated"}


@router.get("/sessions/{session_id}/export")
async def export_artifacts(session_id: str):
    """Downloads all approved artifacts for this session as a grouped Markdown report."""
    artifacts = db.get_session_artifacts(session_id, status="approved")
    title = db.get_session_title(session_id)
    markdown = _build_export_markdown(title, artifacts)
    return PlainTextResponse(
        markdown,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="security-requirements-{session_id[:8]}.md"'}
    )


def _build_export_markdown(title: str, artifacts: list) -> str:
    """Renders a list of approved artifacts as a Markdown document, grouped by
    type in the fixed Asset -> Threat -> Goal -> Req order."""
    lines = [f"# {title}", "", "Approved security artifacts, exported from the Security Requirements Assistant.", ""]
    for art_type in ["Asset", "Threat", "Goal", "Req"]:
        items = [a for a in artifacts if a["type"] == art_type]
        lines.append(f"## {TYPE_HEADINGS[art_type]}")
        lines.append("")
        if items:
            for item in items:
                lines.append(f"- {item['content']}")
        else:
            lines.append("_None approved yet._")
        lines.append("")
    return "\n".join(lines)
