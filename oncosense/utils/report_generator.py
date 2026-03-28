"""
OncoSense - Patient Diagnosis Report Generator
Generates professional PDF reports with diagnosis results, confidence scores, and metrics.
"""

from fpdf import FPDF
from datetime import datetime
import os


class OncoSenseReport(FPDF):
    """Custom PDF report for OncoSense diagnosis."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font('Helvetica', 'B', 20)
        self.set_text_color(75, 0, 130)  # Purple
        self.cell(0, 10, 'OncoSense', align='C', new_x="LMARGIN", new_y="NEXT")
        self.set_font('Helvetica', 'I', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, 'Quantum ML Platform for Early Disease Detection', align='C', new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'OncoSense Report | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")} | Page {self.page_no()}',
                  align='C')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(75, 0, 130)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def key_value(self, key, value, highlight=False):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(60, 60, 60)
        self.cell(70, 7, f'{key}:', new_x="END")
        if highlight:
            self.set_font('Helvetica', 'B', 11)
            if 'Malignant' in str(value):
                self.set_text_color(220, 50, 50)
            elif 'Benign' in str(value):
                self.set_text_color(50, 180, 50)
            else:
                self.set_text_color(75, 0, 130)
        else:
            self.set_font('Helvetica', '', 10)
            self.set_text_color(40, 40, 40)
        self.cell(0, 7, str(value), new_x="LMARGIN", new_y="NEXT")


def generate_report(diagnosis_result, metrics_comparison, patient_features=None, output_path="diagnosis_report.pdf"):
    """
    Generate a patient diagnosis PDF report.

    Parameters:
        diagnosis_result: dict with prediction, label, confidence, etc.
        metrics_comparison: dict of model metrics for comparison
        patient_features: dict of input features (optional)
        output_path: path to save PDF
    """
    pdf = OncoSenseReport()
    pdf.add_page()

    # === DIAGNOSIS RESULT ===
    pdf.section_title('Diagnosis Result')

    # Highlight box
    pdf.set_fill_color(245, 245, 255)
    pdf.set_draw_color(75, 0, 130)
    y_start = pdf.get_y()
    pdf.rect(10, y_start, 190, 35, style='DF')
    pdf.set_xy(15, y_start + 5)

    pdf.set_font('Helvetica', 'B', 18)
    label = diagnosis_result.get('label', 'Unknown')
    if label == 'Malignant':
        pdf.set_text_color(220, 50, 50)
    else:
        pdf.set_text_color(50, 180, 50)
    pdf.cell(0, 10, f'Classification: {label}', new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(15)

    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(75, 0, 130)
    confidence = diagnosis_result.get('confidence', 0)
    pdf.cell(0, 8, f'Confidence Score: {confidence:.2%}', new_x="LMARGIN", new_y="NEXT")
    pdf.set_x(15)

    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f'Model: Quantum Kernel SVM (ZZFeatureMap + Fidelity Quantum Kernel)', new_x="LMARGIN", new_y="NEXT")

    pdf.ln(10)

    # === PROBABILITY BREAKDOWN ===
    pdf.section_title('Probability Breakdown')
    pdf.key_value('Probability of Malignant', f'{diagnosis_result.get("prob_malignant", 0):.4f}')
    pdf.key_value('Probability of Benign', f'{diagnosis_result.get("prob_benign", 0):.4f}')
    pdf.ln(5)

    # === PATIENT FEATURES (if provided) ===
    if patient_features:
        pdf.section_title('Patient Biopsy Data (Input Features)')
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(40, 40, 40)

        col_width = 62
        row_height = 6
        items = list(patient_features.items())

        for i in range(0, len(items), 3):
            batch = items[i:i+3]
            for key, val in batch:
                short_key = key.replace('mean ', '').replace('worst ', 'w.').replace('error ', 'e.')
                pdf.set_font('Helvetica', 'B', 8)
                pdf.cell(35, row_height, short_key[:20], border=0)
                pdf.set_font('Helvetica', '', 8)
                pdf.cell(col_width - 35, row_height, f'{val:.4f}', border=0)
            pdf.ln(row_height)
        pdf.ln(5)

    # === MODEL PERFORMANCE COMPARISON ===
    pdf.section_title('Model Performance Comparison')

    # Table header
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(75, 0, 130)
    pdf.set_text_color(255, 255, 255)
    cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
    widths = [50, 25, 25, 25, 25, 30]
    for col, w in zip(cols, widths):
        pdf.cell(w, 8, col, border=1, fill=True, align='C')
    pdf.ln()

    # Table rows
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(40, 40, 40)

    for model_name, m in metrics_comparison.items():
        row_data = [
            model_name,
            f'{m["accuracy"]:.4f}',
            f'{m["precision"]:.4f}',
            f'{m["recall"]:.4f}',
            f'{m["f1_score"]:.4f}',
            f'{m["auc_roc"]:.4f}',
        ]
        is_quantum = 'Quantum' in model_name
        if is_quantum:
            pdf.set_fill_color(240, 235, 255)
            pdf.set_font('Helvetica', 'B', 9)
        else:
            pdf.set_fill_color(255, 255, 255)
            pdf.set_font('Helvetica', '', 9)

        for val, w in zip(row_data, widths):
            pdf.cell(w, 7, val, border=1, fill=True, align='C')
        pdf.ln()

    pdf.ln(8)

    # === DISCLAIMER ===
    pdf.section_title('Important Disclaimer')
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 5,
        'This report is generated by OncoSense, an experimental quantum machine learning platform. '
        'The diagnosis provided is based on computational analysis of biopsy features and should NOT '
        'be used as the sole basis for medical decisions. Always consult a qualified oncologist or '
        'medical professional for final diagnosis and treatment planning. '
        'Model trained on UCI Wisconsin Breast Cancer Dataset (569 samples).'
    )

    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(75, 0, 130)
    pdf.cell(0, 8, 'OncoSense - Where Quantum Meets Care', align='C')

    # Save
    pdf.output(output_path)
    return output_path
