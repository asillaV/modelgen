"""Simple GUI for modelgen using customtkinter (with tkinter fallback)."""

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Optional

# Try to import customtkinter, fallback to standard tkinter
try:
    import customtkinter as ctk

    CTK_AVAILABLE = True
except ImportError:
    CTK_AVAILABLE = False

from modelgen import logging_setup
from modelgen.io.loaders import load_csv
from modelgen.io.schema import infer_column_types, select_x_axis
from modelgen.ml.artifacts import create_full_report, save_all_artifacts
from modelgen.ml.evaluate import evaluate_model
from modelgen.ml.pipelines import prepare_features, prepare_target
from modelgen.ml.train import get_feature_importance, predict, train_model
from modelgen.viz.plots import create_report_figure, save_report_figure


class ModelGenApp:
    """Main GUI application for modelgen."""

    def __init__(self, root):
        """Initialize the GUI application.

        Args:
            root: Root window (Tk or CTk).
        """
        self.root = root
        self.root.title("ModelGen - CSV Trainer")
        self.root.geometry("800x900")

        # State variables
        self.train_path: Optional[Path] = None
        self.test_path: Optional[Path] = None
        self.df_train = None
        self.schema = None

        # Setup logging to capture in GUI
        logging_setup.setup_logging(level="INFO")

        # Build UI
        self.create_widgets()

    def create_widgets(self):
        """Create and layout all GUI widgets."""
        # Dataset Section
        dataset_frame = self.create_frame(self.root, "1. Dataset Selection")
        dataset_frame.pack(fill="x", padx=10, pady=5)

        self.train_btn = self.create_button(
            dataset_frame, "Select Training CSV", self.select_train_file
        )
        self.train_btn.pack(pady=5)

        self.train_label = self.create_label(dataset_frame, "No file selected")
        self.train_label.pack()

        # Task & Columns Section
        config_frame = self.create_frame(self.root, "2. Task & Columns")
        config_frame.pack(fill="x", padx=10, pady=5)

        # Task selection
        task_row = tk.Frame(config_frame)
        task_row.pack(fill="x", pady=5)
        self.create_label(task_row, "Task:").pack(side="left", padx=5)
        self.task_var = tk.StringVar(value="regression")
        task_reg = tk.Radiobutton(
            task_row, text="Regression", variable=self.task_var, value="regression"
        )
        task_reg.pack(side="left")
        task_cla = tk.Radiobutton(
            task_row, text="Classification", variable=self.task_var, value="classification"
        )
        task_cla.pack(side="left")

        # Load columns button
        self.load_cols_btn = self.create_button(
            config_frame, "Load Columns", self.load_columns
        )
        self.load_cols_btn.pack(pady=5)

        # Target selection
        target_row = tk.Frame(config_frame)
        target_row.pack(fill="x", pady=5)
        self.create_label(target_row, "Target:").pack(side="left", padx=5)
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(target_row, textvariable=self.target_var, state="readonly")
        self.target_combo.pack(side="left", fill="x", expand=True, padx=5)

        # Features selection
        self.create_label(config_frame, "Features (select multiple):").pack(pady=5)
        self.features_listbox = tk.Listbox(
            config_frame, selectmode="multiple", height=6
        )
        self.features_listbox.pack(fill="both", expand=True, padx=5, pady=5)

        # Parameters Section
        params_frame = self.create_frame(self.root, "3. Parameters")
        params_frame.pack(fill="x", padx=10, pady=5)

        # N estimators
        nest_row = tk.Frame(params_frame)
        nest_row.pack(fill="x", pady=5)
        self.create_label(nest_row, "N. Estimators:").pack(side="left", padx=5)
        self.n_est_var = tk.IntVar(value=100)
        tk.Spinbox(nest_row, from_=50, to=500, increment=10, textvariable=self.n_est_var, width=10).pack(side="left")

        # Max depth
        depth_row = tk.Frame(params_frame)
        depth_row.pack(fill="x", pady=5)
        self.create_label(depth_row, "Max Depth (0=None):").pack(side="left", padx=5)
        self.max_depth_var = tk.IntVar(value=0)
        tk.Spinbox(depth_row, from_=0, to=20, textvariable=self.max_depth_var, width=10).pack(side="left")

        # Run Section
        run_frame = self.create_frame(self.root, "4. Train & Evaluate")
        run_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.run_btn = self.create_button(run_frame, "Train Model", self.run_training)
        self.run_btn.pack(pady=10)

        # Log output
        self.log_text = scrolledtext.ScrolledText(run_frame, height=15, wrap=tk.WORD)
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

    def create_frame(self, parent, title):
        """Create a labeled frame."""
        if CTK_AVAILABLE:
            frame = ctk.CTkFrame(parent)
            ctk.CTkLabel(frame, text=title, font=("Arial", 12, "bold")).pack(
                anchor="w", padx=5, pady=5
            )
        else:
            frame = ttk.LabelFrame(parent, text=title)
        return frame

    def create_button(self, parent, text, command):
        """Create a button."""
        if CTK_AVAILABLE:
            return ctk.CTkButton(parent, text=text, command=command)
        else:
            return tk.Button(parent, text=text, command=command)

    def create_label(self, parent, text):
        """Create a label."""
        if CTK_AVAILABLE:
            return ctk.CTkLabel(parent, text=text)
        else:
            return tk.Label(parent, text=text)

    def select_train_file(self):
        """Open file dialog to select training CSV."""
        file_path = filedialog.askopenfilename(
            title="Select Training CSV", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            self.train_path = Path(file_path)
            self.train_label.configure(text=f"File: {self.train_path.name}")
            self.log(f"Selected training file: {self.train_path}")

    def load_columns(self):
        """Load columns from the selected CSV file."""
        if not self.train_path:
            messagebox.showerror("Error", "Please select a training CSV file first")
            return

        try:
            self.log("Loading CSV file...")
            self.df_train, load_report = load_csv(self.train_path, sample_rows=1000)
            self.schema = infer_column_types(self.df_train)

            # Populate target dropdown
            columns = list(self.df_train.columns)
            self.target_combo["values"] = columns
            if columns:
                self.target_var.set(columns[0])

            # Populate features listbox
            self.features_listbox.delete(0, tk.END)
            for col in columns:
                self.features_listbox.insert(tk.END, col)

            self.log(f"Loaded {load_report.shape[0]} rows, {load_report.shape[1]} columns")
            messagebox.showinfo("Success", "Columns loaded successfully")

        except Exception as e:
            self.log(f"Error loading columns: {e}")
            messagebox.showerror("Error", f"Failed to load columns: {e}")

    def run_training(self):
        """Execute the training workflow."""
        if not self.train_path or self.df_train is None:
            messagebox.showerror("Error", "Please load a CSV file and columns first")
            return

        target = self.target_var.get()
        if not target:
            messagebox.showerror("Error", "Please select a target column")
            return

        # Get selected features
        selected_indices = self.features_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "Please select at least one feature")
            return

        feature_columns = [self.features_listbox.get(i) for i in selected_indices]

        if target in feature_columns:
            messagebox.showerror("Error", "Target column cannot be in features")
            return

        try:
            self.log("=" * 50)
            self.log("Starting training workflow...")
            self.run_btn.configure(state="disabled")

            # Get parameters
            task = self.task_var.get()
            n_estimators = self.n_est_var.get()
            max_depth = self.max_depth_var.get() if self.max_depth_var.get() > 0 else None

            self.log(f"Task: {task}, Target: {target}, Features: {len(feature_columns)}")

            # Prepare data
            X, numeric_features, categorical_features = prepare_features(
                self.df_train,
                feature_columns,
                self.schema.numeric_columns,
                self.schema.categorical_columns,
            )
            y = prepare_target(self.df_train, target, task=task)
            X = X.loc[y.index]

            # Train model
            self.log("Training model...")
            pipeline, X_train, X_test, y_train, y_test, training_info = train_model(
                X,
                y,
                numeric_features,
                categorical_features,
                task=task,
                n_estimators=n_estimators,
                max_depth=max_depth,
            )

            # Predict
            y_pred = predict(pipeline, X_test)

            # Evaluate
            self.log("Evaluating model...")
            metrics = evaluate_model(y_test, y_pred, task=task)

            # Create report
            report_data = create_full_report(
                training_info, metrics, feature_columns, target
            )

            # Save artifacts
            output_dir = Path("output")
            artifact_paths = save_all_artifacts(pipeline, y_pred, report_data, output_dir)

            # Create figure if regression
            if task == "regression":
                importance_df = get_feature_importance(pipeline, feature_columns)
                x_axis_data = select_x_axis(self.df_train.loc[X_test.index], None, self.schema)
                metrics_text = f"R²: {metrics['r2']:.3f}, MSE: {metrics['mse']:.2f}"
                fig = create_report_figure(
                    y_test, y_pred, importance_df, x_axis=x_axis_data, metrics_text=metrics_text
                )
                save_report_figure(fig, output_dir)

            # Display results
            self.log("\n" + "=" * 50)
            self.log("TRAINING RESULTS:")
            self.log("=" * 50)
            if task == "regression":
                self.log(f"R² Score: {metrics['r2']:.4f}")
                self.log(f"MSE: {metrics['mse']:.4f}")
                self.log(f"MAE: {metrics['mae']:.4f}")
            else:
                self.log(f"Accuracy: {metrics['accuracy']:.4f}")
                self.log(f"F1-Macro: {metrics['f1_macro']:.4f}")
            self.log("=" * 50)
            self.log(f"\nArtifacts saved to: {output_dir}")

            messagebox.showinfo("Success", "Training completed successfully!")

        except Exception as e:
            self.log(f"\nError during training: {e}")
            messagebox.showerror("Error", f"Training failed: {e}")

        finally:
            self.run_btn.configure(state="normal")

    def log(self, message):
        """Append message to log text area."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()


def main():
    """Main entry point for the GUI application."""
    if CTK_AVAILABLE:
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        root = ctk.CTk()
    else:
        root = tk.Tk()

    app = ModelGenApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
