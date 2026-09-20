"""
CMB Review
PDF Report Generator
"""

from io import BytesIO
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)


def generate_pdf_report(
    filename,
    detections,
    severity,
    mean_confidence,
    review_count
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]

    title_style.alignment = TA_CENTER

    heading_style = styles["Heading2"]

    body_style = styles["BodyText"]

    story = []

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "CMB REVIEW",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Neuroimaging Research Dashboard",
            body_style
        )
    )

    story.append(
        Spacer(
            1,
            15
        )
    )

    story.append(
        Paragraph(
            f"<b>Scan:</b> {filename}",
            body_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Generated:</b> "
            f"{datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            body_style
        )
    )

    story.append(
        Spacer(
            1,
            20
        )
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Analysis Summary",
            heading_style
        )
    )

    summary_data = [
        [
            "Metric",
            "Result"
        ],
        [
            "Candidates",
            str(len(detections))
        ],
        [
            "Mean confidence",
            f"{mean_confidence * 100:.1f}%"
        ],
        [
            "Review recommended",
            str(review_count)
        ],
        [
            "Severity",
            severity
        ]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[
            250,
            180
        ]
    )

    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#153A52")
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#D9E1E8")
                ),
                (
                    "BACKGROUND",
                    (0, 1),
                    (-1, -1),
                    colors.HexColor("#F6F8FA")
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                )
            ]
        )
    )

    story.append(
        summary_table
    )

    story.append(
        Spacer(
            1,
            20
        )
    )


    # --------------------------------------------------------
    # CANDIDATES
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Candidate Findings",
            heading_style
        )
    )

    table_data = [
        [
            "ID",
            "X",
            "Y",
            "Z",
            "Confidence",
            "Classification",
            "Status"
        ]
    ]

    for detection in detections:

        x, y, z = detection["center"]

        table_data.append(
            [
                detection["id"],
                str(x),
                str(y),
                str(z),
                f"{detection['confidence'] * 100:.1f}%",
                detection["classification"],
                detection["confidence_label"]
            ]
        )


    if len(table_data) == 1:

        table_data.append(
            [
                "-",
                "-",
                "-",
                "-",
                "-",
                "No detections",
                "-"
            ]
        )


    findings_table = Table(
        table_data,
        repeatRows=1
    )

    findings_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor("#153A52")
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#D9E1E8")
                ),
                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE"
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                )
            ]
        )
    )

    story.append(
        findings_table
    )

    story.append(
        Spacer(
            1,
            25
        )
    )


    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "<b>Research-use notice:</b> "
            "This software is a research prototype and is not a medical "
            "diagnostic device. Results require qualified clinical review "
            "and should not be used as a standalone diagnostic decision.",
            body_style
        )
    )

    document.build(
        story
    )

    buffer.seek(0)

    return buffer.getvalue()