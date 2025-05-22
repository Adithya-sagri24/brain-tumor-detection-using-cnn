import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import tensorflow as tf
import os
import pygame
import random
import string
import csv
from datetime import datetime
import webbrowser
import pandas as pd
from dateutil.parser import parse

from utils.image_utils import preprocess_image  # type: ignore
from utils.pdf_export import generate_pdf  # type: ignore

MODEL_PATH = "model/brain_tumor_detection_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_LABELS = [
    "Glioma Tumor",
    "Meningioma Tumor",
    "Pituitary Tumor",
    "No Tumor",
    "Other Tumor"
]

pygame.mixer.init()

def generate_patient_id(length=8):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

class NeuroCareApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NEUROCARE - Brain Tumor Detection")
        self.root.geometry("900x680")
        self.root.configure(bg="#F8F4EA")

        self.image_path = None
        self.prediction_label = None
        self.confidence = None

        self.bg_canvas = tk.Canvas(self.root, width=900, height=680, highlightthickness=0)
        self.bg_canvas.pack(fill="both", expand=True)
        self.draw_gradient_background()

        self.setup_ui()

    def draw_gradient_background(self):
        for i in range(0, 900, 2):
            self.bg_canvas.create_line(i, 0, i, 680, fill="#F8F4EA")

    def setup_ui(self):
        title = tk.Label(self.root, text="NEUROCARE", font=("Georgia", 38, "bold"),
                         bg="#F8F4EA", fg="#7A5230")
        self.bg_canvas.create_window(450, 80, window=title)

        content_frame = tk.Frame(self.root, bg="#F8F4EA")
        self.bg_canvas.create_window(450, 280, window=content_frame)

        self.img_frame = tk.Frame(content_frame, bg="#7A5230", bd=2, relief="solid")
        self.img_frame.grid(row=0, column=0, padx=25, pady=10)

        self.img_canvas = tk.Canvas(self.img_frame, width=300, height=300, bg="#F5F2E9", bd=0, highlightthickness=0)
        self.img_canvas.pack()

        self.pred_frame = tk.Frame(self.root, bg="#F8F4EA", bd=2, relief="groove", padx=12, pady=10)
        self.bg_canvas.create_window(450, 510, window=self.pred_frame)

        self.pred_label = tk.Label(self.pred_frame, text="Tumor Type:   —", font=("Helvetica", 16, "bold"),
                                   bg="#F8F4EA", fg="#5A3E2B")
        self.pred_label.pack()

        self.patient_frame = tk.Frame(self.root, bg="#F8F4EA")
        self.bg_canvas.create_window(450, 580, window=self.patient_frame)

        tk.Label(self.patient_frame, text="Name:", font=("Helvetica", 12), bg="#F8F4EA", fg="#5A3E2B").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.name_entry = tk.Entry(self.patient_frame, font=("Helvetica", 12), width=15)
        self.name_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(self.patient_frame, text="DOB:", font=("Helvetica", 12), bg="#F8F4EA", fg="#5A3E2B").grid(row=0, column=2, sticky="e", padx=5, pady=5)
        self.dob_entry = tk.Entry(self.patient_frame, font=("Helvetica", 12), width=15)
        self.dob_entry.grid(row=0, column=3, padx=5, pady=5)
        self.dob_entry.insert(0, "DD-MM-YYYY")

        tk.Label(self.patient_frame, text="Gender:", font=("Helvetica", 12), bg="#F8F4EA", fg="#5A3E2B").grid(row=0, column=4, sticky="e", padx=5, pady=5)
        self.gender_var = tk.StringVar()
        self.gender_menu = ttk.Combobox(self.patient_frame, textvariable=self.gender_var,
                                        values=["Male", "Female", "Other"], font=("Helvetica", 12), width=12, state="readonly")
        self.gender_menu.grid(row=0, column=5, padx=5, pady=5)
        self.gender_menu.current(0)

        self.patient_id = generate_patient_id()
        tk.Label(self.patient_frame, text="Patient ID:", font=("Helvetica", 12), bg="#F8F4EA", fg="#5A3E2B").grid(row=0, column=6, sticky="e", padx=5, pady=5)
        self.patient_id_label = tk.Label(self.patient_frame, text=self.patient_id, font=("Helvetica", 12, "bold"),
                                         bg="#F8F4EA", fg="#7A5230", width=12)
        self.patient_id_label.grid(row=0, column=7, padx=5, pady=5)

        btn_frame = tk.Frame(self.root, bg="#F8F4EA")
        self.bg_canvas.create_window(450, 630, window=btn_frame)

        self.upload_btn = tk.Button(btn_frame, text="Upload MRI", font=("Helvetica", 13, "bold"),
                                    bg="#A47551", fg="white", relief="flat", padx=20, pady=10,
                                    command=self.load_image)
        self.upload_btn.grid(row=0, column=0, padx=15)

        self.pdf_btn = tk.Button(btn_frame, text="Export Report", font=("Helvetica", 13, "bold"),
                                 bg="#A47551", fg="white", relief="flat", padx=20, pady=10,
                                 state=tk.DISABLED, command=self.generate_report)
        self.pdf_btn.grid(row=0, column=1, padx=15)

        self.view_history_btn = tk.Button(btn_frame, text="View History", font=("Helvetica", 13, "bold"),
                                          bg="#A47551", fg="white", relief="flat", padx=20, pady=10,
                                          command=self.view_history)
        self.view_history_btn.grid(row=0, column=2, padx=15)

    def load_image(self):
        filetypes = [("Image files", "*.jpg *.jpeg *.png")]
        path = filedialog.askopenfilename(title="Select MRI Image", filetypes=filetypes)
        if not path:
            return

        self.image_path = path

        img = Image.open(self.image_path)
        img = img.resize((300, 300))
        self.img_tk = ImageTk.PhotoImage(img)
        self.img_canvas.delete("all")
        self.img_canvas.create_image(150, 150, image=self.img_tk)

        self.predict_tumor()

    def predict_tumor(self):
        if not self.image_path:
            return

        img_array = preprocess_image(self.image_path)
        preds = model.predict(img_array)
        class_idx = preds.argmax(axis=1)[0]
        confidence = round(preds[0][class_idx] * 100, 2)
        pred_label = CLASS_LABELS[class_idx]

        self.prediction_label = pred_label
        self.confidence = confidence
        self.pred_label.config(text=f"Tumor Type: {pred_label} ({confidence}%)")

        if confidence < 70:
            messagebox.showwarning("Low Confidence", "Prediction confidence is below 70%. Please consult a doctor for confirmation.")

        try:
            chime_path = os.path.join(os.path.dirname(__file__), "assets", "619840cogfirestudios_achievement-accomplish-jingle-app-ui.wav")
            pygame.mixer.music.load(chime_path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Sound play error: {e}")

        self.pdf_btn.config(state=tk.NORMAL)

    def generate_report(self):
        if not self.prediction_label or not self.image_path:
            return

        name = self.name_entry.get().strip()
        dob = self.dob_entry.get().strip()
        gender = self.gender_var.get()
        patient_id = self.patient_id

        if not name:
            messagebox.showwarning("Missing Info", "Please enter patient name.")
            return

        if not dob or dob == "DD-MM-YYYY":
            messagebox.showwarning("Missing Info", "Please enter patient DOB.")
            return

        patient_info = {
            "Name": name,
            "DOB": dob,
            "Gender": gender,
            "Patient ID": patient_id
        }

        try:
            output_file = filedialog.asksaveasfilename(defaultextension=".pdf",
                                                       filetypes=[("PDF files", "*.pdf")],
                                                       title="Save PDF Report As")
            if not output_file:
                return

            generate_pdf(
                image_path=self.image_path,
                prediction=f"{self.prediction_label} ({self.confidence}%)",
                output_path=output_file,
                patient_info=patient_info,
                model_version="v1.0"
            )
            messagebox.showinfo("Success", f"PDF report saved:\n{output_file}")
            self.save_patient_history()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate PDF report.\n{str(e)}")

    def save_patient_history(self):
        history_file = "patient_history.csv"
        file_exists = os.path.isfile(history_file)

        with open(history_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["DateTime", "Name", "DOB", "Gender", "Patient ID", "Tumor Type", "Confidence (%)", "Image Path"])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.name_entry.get().strip(),
                self.dob_entry.get().strip(),
                self.gender_var.get(),
                self.patient_id,
                self.prediction_label,
                self.confidence,
                self.image_path
            ])

    def calculate_age(self, dob_str):
        try:
            dob = parse(dob_str)
            today = datetime.now()
            age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            return age
        except:
            return None

    def view_history(self):
        history_file = "patient_history.csv"
        if not os.path.exists(history_file):
            messagebox.showinfo("No History", "No patient history found yet.")
            return

        history_window = tk.Toplevel(self.root)
        history_window.title("Patient History")
        history_window.geometry("1100x500")

        # Filter controls
        filter_frame = tk.Frame(history_window)
        filter_frame.pack(fill="x", padx=10, pady=5)

        # Store references to dynamic widgets
        self.filter_label1 = None
        self.filter_label2 = None

        tk.Label(filter_frame, text="Filter By:").pack(side="left", padx=5)
        
        self.filter_type = tk.StringVar()
        filter_menu = ttk.Combobox(filter_frame, textvariable=self.filter_type,
                                values=["All Records", "By Tumor Type", "By Date", "By Age Range"],
                                state="readonly", width=15)
        filter_menu.pack(side="left", padx=5)
        filter_menu.current(0)

        self.filter_value = tk.StringVar()
        self.filter_entry = tk.Entry(filter_frame, textvariable=self.filter_value, width=20)
        self.filter_entry.pack(side="left", padx=5)
        self.filter_entry.config(state=tk.DISABLED)

        self.filter_value2 = tk.StringVar()
        self.filter_entry2 = tk.Entry(filter_frame, textvariable=self.filter_value2, width=20)
        self.filter_entry2.pack(side="left", padx=5)
        self.filter_entry2.config(state=tk.DISABLED)

        def clear_filter_labels():
            # Remove any existing filter labels
            if hasattr(self, 'filter_label1') and self.filter_label1:
                self.filter_label1.destroy()
            if hasattr(self, 'filter_label2') and self.filter_label2:
                self.filter_label2.destroy()

        def update_filter_fields(*args):
            filter_type = self.filter_type.get()
            
            # Clear previous labels and reset entries
            clear_filter_labels()
            self.filter_entry.config(state=tk.NORMAL)
            self.filter_entry.delete(0, tk.END)
            self.filter_entry2.config(state=tk.NORMAL)
            self.filter_entry2.delete(0, tk.END)

            if filter_type == "All Records":
                self.filter_entry.config(state=tk.DISABLED)
                self.filter_entry2.config(state=tk.DISABLED)
            elif filter_type == "By Tumor Type":
                self.filter_entry2.config(state=tk.DISABLED)
                self.filter_label1 = tk.Label(filter_frame, text="Tumor Type:")
                self.filter_label1.pack(side="left", padx=5, before=self.filter_entry)
            elif filter_type == "By Date":
                self.filter_entry2.config(state=tk.DISABLED)
                self.filter_label1 = tk.Label(filter_frame, text="Date (YYYY-MM-DD):")
                self.filter_label1.pack(side="left", padx=5, before=self.filter_entry)
            elif filter_type == "By Age Range":
                self.filter_label1 = tk.Label(filter_frame, text="Min Age:")
                self.filter_label1.pack(side="left", padx=5, before=self.filter_entry)
                self.filter_label2 = tk.Label(filter_frame, text="Max Age:")
                self.filter_label2.pack(side="left", padx=5, before=self.filter_entry2)

        self.filter_type.trace("w", update_filter_fields)
        
        def apply_filter():
            df = pd.read_csv(history_file)
            filter_type = self.filter_type.get()

            if filter_type == "All Records":
                update_treeview(df)
                return
            elif filter_type == "By Tumor Type":
                tumor_type = self.filter_value.get().strip()
                if tumor_type:
                    df = df[df['Tumor Type'].str.contains(tumor_type, case=False)]
            elif filter_type == "By Date":
                date_str = self.filter_value.get().strip()
                if date_str:
                    df = df[df['DateTime'].str.contains(date_str)]
            elif filter_type == "By Age Range":
                try:
                    min_age = int(self.filter_value.get()) if self.filter_value.get().strip() else 0
                    max_age = int(self.filter_value2.get()) if self.filter_value2.get().strip() else 200
                    
                    # Calculate age for each record
                    df['Age'] = df['DOB'].apply(self.calculate_age)
                    df = df[(df['Age'] >= min_age) & (df['Age'] <= max_age)]
                    df = df.drop(columns=['Age'])
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid age numbers")
                    return

            update_treeview(df)

        tk.Button(filter_frame, text="Apply Filter", command=apply_filter).pack(side="left", padx=10)

        # Treeview for displaying records
        tree = ttk.Treeview(history_window, columns=list(range(8)), show="headings", height=15)
        for idx, col in enumerate(["DateTime", "Name", "DOB", "Gender", "Patient ID", "Tumor Type", "Confidence (%)", "Image Path"]):
            tree.heading(idx, text=col)
            tree.column(idx, width=140)
        tree.pack(fill="both", expand=True, padx=10)

        def update_treeview(df):
            tree.delete(*tree.get_children())
            for row in df.values:
                tree.insert("", "end", values=list(row))

        def delete_selected():
            selected = tree.selection()
            if not selected:
                return

            df = pd.read_csv(history_file)
            selected_index = tree.index(selected[0])
            df.drop(index=selected_index, inplace=True)
            df.to_csv(history_file, index=False)
            update_treeview(df)

        def export_filtered():
            df = pd.read_csv(history_file)
            filter_type = self.filter_type.get()

            if filter_type == "By Tumor Type":
                tumor_type = self.filter_value.get().strip()
                if tumor_type:
                    df = df[df['Tumor Type'].str.contains(tumor_type, case=False)]
            elif filter_type == "By Date":
                date_str = self.filter_value.get().strip()
                if date_str:
                    df = df[df['DateTime'].str.contains(date_str)]
            elif filter_type == "By Age Range":
                try:
                    min_age = int(self.filter_value.get()) if self.filter_value.get().strip() else 0
                    max_age = int(self.filter_value2.get()) if self.filter_value2.get().strip() else 200
                    df['Age'] = df['DOB'].apply(self.calculate_age)
                    df = df[(df['Age'] >= min_age) & (df['Age'] <= max_age)]
                    df = df.drop(columns=['Age'])
                except ValueError:
                    messagebox.showerror("Error", "Please enter valid age numbers")
                    return

            output = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if output:
                df.to_csv(output, index=False)
                messagebox.showinfo("Exported", f"Filtered data exported to:\n{output}")

        button_frame = tk.Frame(history_window)
        button_frame.pack(pady=5)

        tk.Button(button_frame, text="Delete Selected", command=delete_selected).pack(side="left", padx=10)
        tk.Button(button_frame, text="Export Filtered", command=export_filtered).pack(side="left", padx=10)

        # Load initial data
        df = pd.read_csv(history_file)
        update_treeview(df)

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuroCareApp(root)
    root.mainloop()