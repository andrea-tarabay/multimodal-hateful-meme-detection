"""
inspect_dataset_gui.py

Tkinter GUI for browsing meme dataset splits.

Loads train/val/test JSONL files from a dataset root directory and displays
each record's image, text, label, and split in a scrollable list.
Supports filtering by split, label, source, and free-text search on id/text.

Usage:
    python dataset_related/inspect_dataset_gui.py --dataset-root /path/to/dataset
"""

import argparse
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    from PIL import Image, ImageTk

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


def load_jsonl(path, split):
    records = []
    by_rel_path = {}
    if not os.path.exists(path):
        return records, by_rel_path
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_split"] = split
            records.append(record)
            image_path = record.get("image_path")
            if image_path:
                by_rel_path[os.path.normpath(image_path)] = record
    return records, by_rel_path


def load_dataset(dataset_root):
    all_records = []
    by_rel_path = {}
    for split in ("train", "val", "test"):
        path = os.path.join(dataset_root, f"{split}.jsonl")
        records, rel_map = load_jsonl(path, split)
        all_records.extend(records)
        by_rel_path.update(rel_map)
    return all_records, by_rel_path


def find_record(dataset_root, by_rel_path, selected_path):
    try:
        rel_path = os.path.relpath(selected_path, dataset_root)
        rel_path = os.path.normpath(rel_path)
    except Exception:
        rel_path = None

    if rel_path and rel_path in by_rel_path:
        return by_rel_path[rel_path]

    # Fallback: match by basename if needed.
    basename = os.path.basename(selected_path)
    for key, record in by_rel_path.items():
        if os.path.basename(key) == basename:
            return record

    return None


def format_record(record):
    fields = [
        ("id", record.get("id")),
        ("split", record.get("_split")),
        ("image_path", record.get("image_path")),
        ("label", record.get("label")),
        ("source", record.get("source")),
        ("text", record.get("text")),
    ]
    lines = []
    for key, value in fields:
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def build_gui(dataset_root, records, by_rel_path):
    root = tk.Tk()
    root.title("Meme dataset inspector")
    root.geometry("1100x700")

    style = ttk.Style(root)
    theme_name = "clam" if "clam" in style.theme_names() else style.theme_use()
    style.theme_use(theme_name)

    root.configure(bg="#ffffff")

    style.configure("TFrame", background="#ffffff")
    style.configure("Card.TFrame", background="#ffffff")
    style.configure("TLabel", background="#ffffff", foreground="#0f172a", font=("DejaVu Sans", 11))
    style.configure("Title.TLabel", font=("DejaVu Sans", 16, "bold"))
    style.configure("Header.TLabel", font=("DejaVu Sans", 12, "bold"))
    style.configure("Filter.TLabel", background="#ffffff", foreground="#0f172a")
    style.configure("Card.TLabelframe", background="#ffffff", foreground="#0f172a")
    style.configure("Card.TLabelframe.Label", background="#ffffff", foreground="#0f172a", font=("DejaVu Sans", 10, "bold"))
    style.configure(
        "TButton",
        background="#1f2937",
        foreground="#ffffff",
        font=("DejaVu Sans", 11, "bold"),
        padding=(10, 6),
    )
    style.map(
        "TButton",
        background=[("active", "#111827"), ("disabled", "#9ca3af")],
        foreground=[("disabled", "#e5e7eb")],
    )
    style.configure("TEntry", fieldbackground="#ffffff", foreground="#111827", padding=(6, 4))
    style.configure("TCombobox", padding=(6, 4), foreground="#111827")
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", "#ffffff"), ("!readonly", "#ffffff")],
        foreground=[("readonly", "#111827"), ("!readonly", "#111827")],
    )
    style.configure("TScrollbar", background="#ffffff")

    main_frame = ttk.Frame(root, padding=12)
    main_frame.pack(fill=tk.BOTH, expand=True)

    left_frame = ttk.Frame(main_frame)
    left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

    right_frame = ttk.Frame(main_frame)
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    title_label = ttk.Label(left_frame, text="Dataset Viewer", style="Title.TLabel")
    title_label.pack(anchor=tk.W, pady=(0, 8))

    list_label = ttk.Label(left_frame, text="Records", style="Header.TLabel")
    list_label.pack(anchor=tk.W)

    filter_frame = ttk.LabelFrame(left_frame, text="Filters", padding=10, style="Card.TLabelframe")
    filter_frame.pack(fill=tk.X, pady=(6, 8))

    search_var = tk.StringVar()
    split_var = tk.StringVar(value="All")
    label_var = tk.StringVar(value="All")
    source_var = tk.StringVar(value="All")

    ttk.Label(filter_frame, text="Search id/text", style="Filter.TLabel").grid(
        row=0, column=0, sticky=tk.W
    )
    search_entry = ttk.Entry(filter_frame, textvariable=search_var)
    search_entry.grid(row=0, column=1, sticky=tk.EW, padx=(8, 0))

    ttk.Label(filter_frame, text="Split", style="Filter.TLabel").grid(
        row=1, column=0, sticky=tk.W, pady=(6, 0)
    )
    split_combo = ttk.Combobox(filter_frame, textvariable=split_var, state="readonly")
    split_combo.grid(row=1, column=1, sticky=tk.EW, padx=(8, 0), pady=(6, 0))

    ttk.Label(filter_frame, text="Label", style="Filter.TLabel").grid(
        row=2, column=0, sticky=tk.W, pady=(6, 0)
    )
    label_combo = ttk.Combobox(filter_frame, textvariable=label_var, state="readonly")
    label_combo.grid(row=2, column=1, sticky=tk.EW, padx=(8, 0), pady=(6, 0))

    ttk.Label(filter_frame, text="Source", style="Filter.TLabel").grid(
        row=3, column=0, sticky=tk.W, pady=(6, 0)
    )
    source_combo = ttk.Combobox(filter_frame, textvariable=source_var, state="readonly")
    source_combo.grid(row=3, column=1, sticky=tk.EW, padx=(8, 0), pady=(6, 0))

    filter_frame.columnconfigure(1, weight=1)

    list_frame = ttk.Frame(left_frame, style="Card.TFrame", padding=6)
    list_frame.pack(fill=tk.BOTH, expand=True)

    listbox = tk.Listbox(
        list_frame,
        width=40,
        height=30,
        bg="#ffffff",
        fg="#111827",
        selectbackground="#111827",
        selectforeground="#ffffff",
        activestyle="none",
        highlightthickness=1,
        highlightbackground="#e2e8f0",
        relief=tk.FLAT,
    )
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.config(yscrollcommand=scrollbar.set)

    button_frame = ttk.Frame(left_frame)
    button_frame.pack(fill=tk.X, pady=(10, 0))

    info_label = ttk.Label(right_frame, text="Record info", style="Header.TLabel")
    info_label.pack(anchor=tk.W)

    info_frame = ttk.Frame(right_frame, style="Card.TFrame", padding=10)
    info_frame.pack(fill=tk.X)

    info_vars = {
        "id": tk.StringVar(value="-"),
        "split": tk.StringVar(value="-"),
        "label": tk.StringVar(value="-"),
        "source": tk.StringVar(value="-"),
        "image_path": tk.StringVar(value="-"),
    }

    info_pairs = [
        ("Id", "id"),
        ("Split", "split"),
        ("Label", "label"),
        ("Source", "source"),
        ("Image path", "image_path"),
    ]

    for row_index, (label_text, key) in enumerate(info_pairs):
        ttk.Label(info_frame, text=f"{label_text}:").grid(row=row_index, column=0, sticky=tk.W)
        ttk.Label(info_frame, textvariable=info_vars[key]).grid(
            row=row_index, column=1, sticky=tk.W
        )

    info_frame.columnconfigure(1, weight=1)

    ttk.Label(right_frame, text="Text", style="Header.TLabel").pack(anchor=tk.W, pady=(12, 0))
    info_text = tk.Text(
        right_frame,
        height=8,
        wrap=tk.WORD,
        bg="#ffffff",
        fg="#111827",
        highlightthickness=1,
        highlightbackground="#e2e8f0",
        relief=tk.FLAT,
    )
    info_text.pack(fill=tk.BOTH, expand=False)

    image_frame = ttk.Frame(right_frame, style="Card.TFrame", padding=10)
    image_frame.pack(fill=tk.BOTH, expand=True, pady=(12, 0))
    
    image_label = ttk.Label(image_frame)
    image_label.pack(fill=tk.BOTH, expand=True)

    status_var = tk.StringVar(value=f"Loaded {len(records)} records")
    status_bar = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    image_cache = {"tk_image": None, "pil_image": None, "image_path": None, "resize_job": None}

    filtered_records = list(records)

    def render_image():
        if PIL_AVAILABLE and image_cache["pil_image"] is not None:
            width = image_label.winfo_width()
            height = image_label.winfo_height()
            if width <= 1 or height <= 1:
                return
            max_w = max(100, width - 10)
            max_h = max(100, height - 10)
            image = image_cache["pil_image"].copy()
            image.thumbnail((max_w, max_h))
            tk_image = ImageTk.PhotoImage(image)
            image_cache["tk_image"] = tk_image
            image_label.configure(image=tk_image)

    def update_info(record, image_path):
        info_text.delete("1.0", tk.END)
        if record is None:
            for key in info_vars:
                info_vars[key].set("-")
            info_text.insert(tk.END, "No record found for selected image.")
            status_var.set("No matching record")
            image_label.configure(image="")
            image_cache["tk_image"] = None
            image_cache["pil_image"] = None
            image_cache["image_path"] = None
            return

        info_vars["id"].set(record.get("id"))
        info_vars["split"].set(record.get("_split"))
        info_vars["label"].set(record.get("label"))
        info_vars["source"].set(record.get("source"))
        info_vars["image_path"].set(record.get("image_path"))

        info_text.insert(tk.END, record.get("text", ""))
        status_var.set(f"Showing {record.get('id')} ({record.get('_split')})")

        if not image_path:
            return

        if PIL_AVAILABLE:
            try:
                image = Image.open(image_path)
            except Exception as exc:
                messagebox.showerror("Image load error", str(exc))
                return
            image_cache["pil_image"] = image
            image_cache["image_path"] = image_path
            render_image()
        else:
            try:
                tk_image = tk.PhotoImage(file=image_path)
            except Exception as exc:
                messagebox.showerror("Image load error", str(exc))
                return
            image_cache["tk_image"] = tk_image
            image_label.configure(image=tk_image)

    def select_record(event=None):
        selection = listbox.curselection()
        if not selection:
            return
        index = selection[0]
        record = filtered_records[index]
        image_path = record.get("image_path")
        full_path = None
        if image_path:
            full_path = os.path.join(dataset_root, image_path)
        update_info(record, full_path)

    def open_image():
        selected_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")],
        )
        if not selected_path:
            return
        record = find_record(dataset_root, by_rel_path, selected_path)
        update_info(record, selected_path)

    open_button = ttk.Button(button_frame, text="Open image...", command=open_image)
    open_button.pack(fill=tk.X)

    clear_button = ttk.Button(button_frame, text="Clear filters", command=lambda: clear_filters())
    clear_button.pack(fill=tk.X, pady=(6, 0))

    def apply_filters(event=None):
        nonlocal filtered_records
        query = search_var.get().strip().lower()
        split = split_var.get()
        label = label_var.get()
        source = source_var.get()

        filtered_records = []
        for record in records:
            if split != "All" and record.get("_split") != split:
                continue
            if label != "All" and str(record.get("label")) != label:
                continue
            if source != "All" and record.get("source") != source:
                continue
            if query:
                record_id = str(record.get("id", "")).lower()
                text = str(record.get("text", "")).lower()
                if query not in record_id and query not in text:
                    continue
            filtered_records.append(record)

        listbox.delete(0, tk.END)
        for record in filtered_records:
            label_value = record.get("label")
            split_value = record.get("_split")
            record_id = record.get("id")
            listbox.insert(tk.END, f"{split_value} | {record_id} | label={label_value}")
        status_var.set(f"Showing {len(filtered_records)} of {len(records)} records")

    def clear_filters():
        search_var.set("")
        split_var.set("All")
        label_var.set("All")
        source_var.set("All")
        apply_filters()

    splits = ["All"] + sorted({record.get("_split") for record in records if record.get("_split")})
    labels = ["All"] + sorted({str(record.get("label")) for record in records if record.get("label") is not None})
    sources = ["All"] + sorted({record.get("source") for record in records if record.get("source")})
    split_combo["values"] = splits
    label_combo["values"] = labels
    source_combo["values"] = sources

    apply_filters()

    listbox.bind("<<ListboxSelect>>", select_record)
    def clear_selection_on_focus(event):
        try:
            event.widget.selection_clear()
        except tk.TclError:
            pass

    search_entry.bind("<Return>", apply_filters)
    search_entry.bind("<FocusIn>", clear_selection_on_focus)
    split_combo.bind("<FocusIn>", clear_selection_on_focus)
    label_combo.bind("<FocusIn>", clear_selection_on_focus)
    source_combo.bind("<FocusIn>", clear_selection_on_focus)
    split_combo.bind("<<ComboboxSelected>>", apply_filters)
    label_combo.bind("<<ComboboxSelected>>", apply_filters)
    source_combo.bind("<<ComboboxSelected>>", apply_filters)

    def schedule_resize(event=None):
        if image_cache["resize_job"] is not None:
            root.after_cancel(image_cache["resize_job"])
        image_cache["resize_job"] = root.after(150, render_image)

    root.bind("<Configure>", schedule_resize)

    return root


def main():
    parser = argparse.ArgumentParser(description="Inspect merged meme dataset.")
    parser.add_argument(
        "--dataset-root",
        default=os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "meme_dataset")),
        help="Path to dataset folder containing train/val/test jsonl.",
    )
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        raise SystemExit(f"Dataset root not found: {dataset_root}")

    records, by_rel_path = load_dataset(dataset_root)
    if not records:
        raise SystemExit("No records found in train/val/test jsonl.")

    root = build_gui(dataset_root, records, by_rel_path)
    root.mainloop()


if __name__ == "__main__":
    main()
