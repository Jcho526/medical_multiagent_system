"""Output formatters — JSON and PDF report generation."""

import json
from typing import Dict, Any
from pathlib import Path


def to_structured_json(result: Dict[str, Any]) -> str:
    """Convert pipeline result to the simplified structured JSON format."""
    symptom = result.get("symptom_analysis", {})
    risk = result.get("risk_assessment", {})
    rehab = result.get("rehab_plan", {})

    # Extract diagnosis from possible_conditions
    conditions = symptom.get("possible_conditions", [])
    diagnosis = ""
    if conditions:
        top = conditions[0]
        diagnosis = f"{top.get('name', '未知')}（置信度: {top.get('confidence', '未知')}）"
        if len(conditions) > 1:
            others = [c.get("name", "") for c in conditions[1:]]
            diagnosis += f"；需鉴别: {', '.join(others)}"

    # Extract plan steps
    plan_phases = rehab.get("rehab_plan", {}).get("phases", [])
    plan_steps = []
    for phase in plan_phases:
        phase_name = phase.get("phase", "")
        activities = phase.get("activities", [])
        if activities:
            for act in activities:
                plan_steps.append(f"[{phase_name}] {act.get('name', '')} — {act.get('method', '')}")
        else:
            objectives = phase.get("objectives", [])
            plan_steps.append(f"[{phase_name}] 目标: {'; '.join(objectives)}")

    # Add home practice
    home = rehab.get("home_practice", [])
    if home:
        plan_steps.append(f"[家庭练习] {'; '.join(home)}")

    output = {
        "diagnosis": diagnosis or symptom.get("analysis_summary", "分析完成"),
        "risk_level": risk.get("risk_level", "未知"),
        "risk_score": risk.get("risk_score", 0),
        "urgency": risk.get("urgency", ""),
        "plan": plan_steps,
        "precautions": rehab.get("precautions", []),
        "follow_up": rehab.get("follow_up", {}),
    }
    return json.dumps(output, ensure_ascii=False, indent=2)


def to_pdf(result: Dict[str, Any], output_path: str = "report.pdf") -> str:
    """Generate a PDF report (requires reportlab)."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        raise ImportError("reportlab is required for PDF generation. Install: pip install reportlab")

    # Try to register a Chinese font
    font_name = "Helvetica"
    chinese_font_available = False
    try:
        # macOS内置的中文字体
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
        ]
        for fp in font_paths:
            if Path(fp).exists():
                pdfmetrics.registerFont(TTFont("Chinese", fp))
                font_name = "Chinese"
                chinese_font_available = True
                break
    except Exception:
        pass

    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=20,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=14,
    )
    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=10,
        leading=14,
    )

    elements = []
    elements.append(Paragraph("Medical Rehab Agent Report", title_style))
    elements.append(Spacer(1, 10 * mm))

    # Patient info
    patient = result.get("patient", {})
    elements.append(Paragraph("Patient Information", heading_style))
    elements.append(Paragraph(f"Age: {patient.get('age', 'N/A')}", body_style))
    elements.append(Paragraph(f"Symptoms: {patient.get('symptoms', 'N/A')}", body_style))
    elements.append(Paragraph(f"History: {patient.get('history', 'N/A')}", body_style))
    elements.append(Spacer(1, 5 * mm))

    # Use structured JSON for the rest
    structured = json.loads(to_structured_json(result))

    elements.append(Paragraph("Diagnosis", heading_style))
    elements.append(Paragraph(str(structured.get("diagnosis", "N/A")), body_style))
    elements.append(Spacer(1, 5 * mm))

    elements.append(Paragraph("Risk Assessment", heading_style))
    risk_text = f"Level: {structured.get('risk_level', 'N/A')} | Score: {structured.get('risk_score', 'N/A')} | Urgency: {structured.get('urgency', 'N/A')}"
    elements.append(Paragraph(risk_text, body_style))
    elements.append(Spacer(1, 5 * mm))

    elements.append(Paragraph("Rehabilitation Plan", heading_style))
    for step in structured.get("plan", []):
        elements.append(Paragraph(f"- {step}", body_style))
    elements.append(Spacer(1, 5 * mm))

    if structured.get("precautions"):
        elements.append(Paragraph("Precautions", heading_style))
        for p in structured["precautions"]:
            elements.append(Paragraph(f"- {p}", body_style))

    doc.build(elements)
    return output_path
