from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
from PIL import Image
import os
import uuid

def wrap_text(text, max_width, canvas_obj, font_name="Helvetica", font_size=11):
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + " " + word if current_line else word
        if canvas_obj.stringWidth(test_line, font_name, font_size) <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    return lines

def generate_pdf(
    image_path,
    prediction,
    output_path="tumor_report.pdf",
    patient_info=None,
    model_version="v1.0",
    doctor_notes="N/A",
    confidence=0.0
):
    # Setup canvas
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    margin = 50
    line_height = 20
    y_pos = height - margin

    # Logo
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        c.drawImage(logo_path, margin, height - 80, width=100, height=50, preserveAspectRatio=True)

    # Title
    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin + 120, height - margin, "NEUROCARE - Tumor Report")

    # Date & Time
    c.setFont("Helvetica", 12)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(margin, height - margin - 30, f"Generated on: {now}")

    # Report ID
    report_id = str(uuid.uuid4())[:8]
    c.drawString(margin, height - margin - 50, f"Report ID: {report_id}")

    # Separator line
    c.setLineWidth(1)
    c.line(margin, y_pos - 70, width - margin, y_pos - 70)
    y_pos -= 90

    # Patient Info
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y_pos, "Patient Information:")
    y_pos -= line_height

    c.setFont("Helvetica", 12)
    if patient_info:
        for key, value in patient_info.items():
            c.drawString(margin + 20, y_pos, f"{key}: {value}")
            y_pos -= line_height
    else:
        c.drawString(margin + 20, y_pos, "No patient information provided.")
        y_pos -= line_height

    # Prediction Section
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y_pos - 10, f"Tumor Prediction: {prediction}")
    y_pos -= line_height + 10

    c.setFont("Helvetica", 12)
    formatted_conf = f"{confidence:.2f}% confidence"
    c.drawString(margin + 20, y_pos, f"Model Confidence: {formatted_conf}")
    y_pos -= line_height + 10

    # Insert Image
    try:
        img = Image.open(image_path)
        img.thumbnail((300, 300))
        temp_img_path = "temp_preview.jpg"
        img.save(temp_img_path)
        c.drawImage(temp_img_path, margin, y_pos - 220, width=200, height=200)
        os.remove(temp_img_path)
    except Exception as e:
        c.drawString(margin, y_pos - 15, "Image preview error.")
    y_pos -= 240

    # Tumor Descriptions
    tumor_label = prediction.split(" (")[0]
    tumor_descriptions = {
        "Glioma Tumor": (
            "Gliomas are a diverse group of brain tumors originating from glial cells, which support and protect neurons. "
            "They are classified by cell type (astrocytoma, oligodendroglioma, ependymoma) and grade (I-IV) based on aggressiveness. "
            "Symptoms may include headaches, seizures, cognitive or personality changes, and neurological deficits depending on tumor location. "
            "Treatment often involves a combination of surgery, radiation therapy, and chemotherapy. Prognosis varies by tumor type and grade; "
            "high-grade gliomas (such as glioblastoma) are more aggressive and challenging to treat."
        ),
        "Meningioma Tumor": (
            "Meningiomas develop from the meninges, the protective membranes covering the brain and spinal cord. "
            "Most meningiomas are benign (non-cancerous) and slow-growing, but some can be atypical or malignant. "
            "Symptoms depend on tumor size and location, and may include headaches, vision problems, seizures, or weakness. "
            "Treatment options include observation for small, asymptomatic tumors, surgical removal, and sometimes radiation therapy. "
            "Recurrence is possible, especially for atypical or malignant meningiomas."
        ),
        "Pituitary Tumor": (
            "Pituitary tumors arise from the pituitary gland, a small gland at the base of the brain responsible for hormone production. "
            "They can be functioning (hormone-secreting) or non-functioning. Symptoms may include hormonal imbalances, vision changes, headaches, "
            "and fatigue. Common types include prolactinomas, growth hormone-secreting tumors, and ACTH-secreting tumors. "
            "Treatment may involve medications to control hormone levels, surgery (often via a transsphenoidal approach), and/or radiation therapy."
        ),
        "No Tumor": (
            "No abnormal tumor structures were identified in the submitted scan. The brain appears within normal limits for the evaluated regions. "
            "If symptoms persist, further clinical evaluation may be warranted to rule out non-tumorous causes."
        ),
        "Other Tumor": (
            "A brain lesion was detected, but its type could not be classified by the current model. "
            "This may represent a rare tumor, metastasis, or non-tumorous abnormality. "
            "Further diagnostic imaging (such as MRI with contrast), clinical correlation, and possibly biopsy are recommended for definitive diagnosis."
        )
    }
    full_desc = tumor_descriptions.get(tumor_label, "No detailed description available.")

    # Draw wrapped Tumor Description
    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin, y_pos, "Tumor Description:")
    y_pos -= line_height
    c.setFont("Helvetica", 11)
    desc_lines = wrap_text(full_desc, width - 2 * margin, c, font_size=11)
    for line in desc_lines:
        c.drawString(margin + 10, y_pos, line)
        y_pos -= line_height

    y_pos -= 10

    # Clinical Recommendations
    full_recommendation = (
        "Seek neurosurgical evaluation. Observation or surgical removal may be required "
        "based on tumor size, type, and symptoms."
    )
    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin, y_pos, "Clinical Recommendations:")
    y_pos -= line_height
    c.setFont("Helvetica", 11)
    rec_lines = wrap_text(full_recommendation, width - 2 * margin, c, font_size=11)
    for line in rec_lines:
        c.drawString(margin + 10, y_pos, line)
        y_pos -= line_height

    y_pos -= 10

    # Doctor's Notes
    c.setFont("Helvetica-Bold", 13)
    c.drawString(margin, y_pos, "Doctor's Notes / Diagnosis:")
    y_pos -= line_height
    c.setFont("Helvetica", 11)
    doctor_lines = wrap_text(doctor_notes, width - 2 * margin, c, font_size=11)
    for line in doctor_lines:
        c.drawString(margin + 10, y_pos, line)
        y_pos -= line_height

    y_pos -= 10

    # Disclaimer
    disclaimer = (
        "Disclaimer: This report is generated using an AI-based tool and is not a substitute "
        "for professional medical advice. Always consult a certified healthcare provider."
    )
    c.setFont("Helvetica-Oblique", 9)
    disclaimer_lines = wrap_text(disclaimer, width - 2 * margin, c, font_name="Helvetica-Oblique", font_size=9)
    for line in disclaimer_lines:
        c.drawString(margin, y_pos, line)
        y_pos -= line_height

    # Footer
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(margin, 40, f"Model Version: {model_version}")
    c.drawString(margin, 25, f"Image file: {os.path.basename(image_path)}")

    footer_text = "NEUROCARE | Brain Tumor Detection System"
    footer_width = c.stringWidth(footer_text, "Helvetica-Oblique", 10)
    c.drawString(width - margin - footer_width, 25, footer_text)

    # Page Number
    c.setFont("Helvetica", 9)
    c.drawString(width - margin, 10, f"Page 1")

    # Save
    c.showPage()
    c.save()
