# for building the .exe
import sys
from pathlib import Path

def resource_path(rel_path: str) -> str:
    base = getattr(sys, "_MEIPASS", Path(__file__).parent)
    return str(Path(base, rel_path))

import tkinter as tk
from tkinter import ttk
import pandas as pd
from PIL import Image, ImageTk, Image
import os, string, threading
from poke_team_optimizer import optimize_team

# config
SPRITE_DIR = resource_path("sprites")
SPRITE_SIZE = (64, 64)
NAME_WIDTH  = 18
NUM_POKEMON = 6

# helpers
def normalize_name(name: str) -> str:
    return name.translate(str.maketrans('', '', string.punctuation)).replace(' ', '').lower()

def keynorm(s: str) -> str:
    return ''.join(ch for ch in str(s).lower() if ch.isalnum())

def fmt_stat(val):
    if pd.isna(val): return "—"
    s = str(val).strip()
    return "—" if s.lower() in {"nan", "none", "null", ""} else s

# load data
df_pokemon = pd.read_csv(resource_path("pokemon.csv")).drop_duplicates(subset=["Pokemon Name"], keep="first")
for col in ["Pokemon Name","Hidden Ability","Legendary Type"]:
    if col in df_pokemon.columns:
        df_pokemon[col] = df_pokemon[col].astype(str).str.strip().str.strip('"').str.strip("'")
df_pokemon["__norm_name__"] = df_pokemon["Pokemon Name"].apply(normalize_name)

# resolve issues with stat columns
col_index = {keynorm(c): c for c in df_pokemon.columns}
def resolve(names):
    for n in names:
        k = keynorm(n)
        if k in col_index: return col_index[k]
    return None

HP_COL             = resolve(["HP","Health","Health Stat"])
ATTACK_COL         = resolve(["Attack","Atk","Attack Stat"])
DEFENSE_COL        = resolve(["Defense","Def","Defense Stat"])
SP_ATTACK_COL      = resolve(["Special Attack","Sp. Atk","Special Attack Stat","Sp Atk"])
SP_DEFENSE_COL     = resolve(["Special Defense","Sp. Def","Special Defense Stat","Sp Def"])
SPEED_COL          = resolve(["Speed","Spe","Speed Stat"])
HIDDEN_ABILITY_COL = resolve(["Hidden Ability","HiddenAbility"])  # kept for completeness; not displayed

STAT_COLS = {
    "HP": HP_COL,
    "Attack": ATTACK_COL,
    "Defense": DEFENSE_COL,
    "Special Attack": SP_ATTACK_COL,
    "Special Defense": SP_DEFENSE_COL,
    "Speed": SPEED_COL,
    "Hidden Ability": HIDDEN_ABILITY_COL,
}

for key, col in STAT_COLS.items():
    if col and key != "Hidden Ability":
        df_pokemon[col] = pd.to_numeric(df_pokemon[col], errors="coerce")

# stats display, no hidden ability rn
selected_stats = ["HP", "Attack", "Defense", "Special Attack", "Special Defense", "Speed"]
friendly_stats = ["Score"] + selected_stats

def get_pokemon_row(name: str) -> pd.Series:
    norm = normalize_name(name)
    hit = df_pokemon[df_pokemon["__norm_name__"] == norm]
    if not hit.empty: return hit.iloc[0]
    hit = df_pokemon[df_pokemon["Pokemon Name"] == name]
    return hit.iloc[0] if not hit.empty else pd.Series()

# sprites
all_pokemon_names = df_pokemon["Pokemon Name"].tolist()
sprite_dict = {}
for n in all_pokemon_names:
    p = os.path.join(SPRITE_DIR, normalize_name(n) + ".png")
    sprite_dict[n] = p if os.path.isfile(p) else None

# gui base
root = tk.Tk()
root.title("Pokémon Optimizer GUI")
empty_sprite = ImageTk.PhotoImage(Image.new("RGBA", SPRITE_SIZE, (255,255,255,0)))

main = ttk.Frame(root)
main.pack(side="top", anchor="nw", padx=10, pady=10)

# compact column widths in stats grids
LABEL_MIN = 84
VALUE_MIN = 110
COL_SIZES = {0: LABEL_MIN, 1: VALUE_MIN, 2: LABEL_MIN, 3: VALUE_MIN}  # two label/value pairs per row

# friendly team
friendly_frame = ttk.LabelFrame(main, text="Friendly Team")
friendly_frame.pack(side="left", anchor="nw")  # no stretch

friendly_rows = []
for i in range(NUM_POKEMON):
    row = {}

    pic = ttk.Label(friendly_frame, image=empty_sprite)
    pic.grid(row=i, column=0, padx=2, pady=4, sticky="w")
    row["sprite"] = pic

    name = ttk.Label(friendly_frame, text="Empty", width=NAME_WIDTH, anchor="w")
    name.grid(row=i, column=1, padx=6, pady=4, sticky="w")
    row["name"] = name

    stats_frame = ttk.Frame(friendly_frame)
    stats_frame.grid(row=i, column=2, padx=4, pady=4, sticky="nw")
    for c, ms in COL_SIZES.items():
        stats_frame.grid_columnconfigure(c, weight=0, minsize=ms)

    labels = []
    ttk.Label(stats_frame, text="Score:", anchor="e").grid(row=0, column=0, sticky="e", padx=2)
    score_val = ttk.Label(stats_frame, text="", anchor="w")
    score_val.grid(row=0, column=1, sticky="w", padx=4)
    labels.append(score_val)

    for idx, stat in enumerate(selected_stats):
        col = idx // 3
        r   = idx % 3 + 1
        ttk.Label(stats_frame, text=f"{stat}:", anchor="e").grid(row=r, column=2*col, sticky="e", padx=2)
        v = ttk.Label(stats_frame, text="", anchor="w")
        v.grid(row=r, column=2*col+1, sticky="w", padx=4)
        labels.append(v)

    row["stats_labels"] = labels
    friendly_rows.append(row)

# opponent team
class AutocompleteEntry(tk.Frame):
    def __init__(self, master, names, img_label, stats_frame, width=NAME_WIDTH):
        super().__init__(master)
        self.names, self.img_label = names, img_label
        self.stats_labels = []

        for c, ms in COL_SIZES.items():
            stats_frame.grid_columnconfigure(c, weight=0, minsize=ms)

        for idx, stat in enumerate(selected_stats):
            col = idx // 3
            r   = idx % 3
            ttk.Label(stats_frame, text=f"{stat}:", anchor="e").grid(row=r, column=2*col, sticky="e", padx=2)
            v = ttk.Label(stats_frame, text="", anchor="w")
            v.grid(row=r, column=2*col+1, sticky="w", padx=4)
            self.stats_labels.append(v)

        self.var = tk.StringVar()
        self.entry = tk.Entry(self, textvariable=self.var, width=width, relief="solid", bd=1)
        self.entry.pack(side="left", fill="x", expand=True)
        self.var.trace_add("write", self.on_change)

        self.arrow_btn = tk.Button(self, text="▾", relief="flat", bd=0, padx=3, pady=0, command=self.show_all)
        self.arrow_btn.pack(side="right", fill="y")

        self.clear_btn = tk.Button(self, text="✕", relief="flat", bd=0, padx=3, pady=0,
                                   command=self.clear_entry, fg="red")
        self.clear_btn.pack(side="right", fill="y")

        self.dropdown = tk.Toplevel(master)
        self.dropdown.withdraw()
        self.dropdown.overrideredirect(True)
        self.dropdown.attributes("-topmost", True)
        self.dropdown.config(bd=1, relief="solid")

        self.listbox = tk.Listbox(self.dropdown, height=6, activestyle="dotbox",
                                  selectbackground="#3399FF", selectforeground="white", bd=0)
        self.listbox.pack(fill="both", expand=True, padx=2, pady=2)
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        self.listbox.bind("<Motion>", self._on_hover)
        master.bind_all("<Button-1>", self._click_outside, add="+")

    def show_all(self):
        self.var.set("")
        self.on_change()

    def on_change(self, *args):
        typed = self.var.get().lower()
        matches = [n for n in self.names if typed in n.lower()]
        if matches:
            self.listbox.delete(0, tk.END)
            for n in matches: self.listbox.insert(tk.END, n)
            x = self.entry.winfo_rootx()
            y = self.entry.winfo_rooty() + self.entry.winfo_height()
            if y + self.listbox.winfo_reqheight() > self.entry.winfo_screenheight():
                y = self.entry.winfo_rooty() - self.listbox.winfo_reqheight()
            width = max(self.entry.winfo_width(), 200)
            self.dropdown.geometry(f"{width}x{self.listbox.winfo_reqheight()}+{x}+{y}")
            self.dropdown.deiconify(); self.dropdown.lift()
        else:
            self.dropdown.withdraw()

    def on_select(self, event):
        if not self.listbox.curselection(): return
        sel = self.listbox.get(self.listbox.curselection()[0])
        self.var.set(sel)
        self.update_preview(sel)
        self.entry.master.focus_set()
        self.dropdown.withdraw()

    def clear_entry(self):
        self.var.set("")
        self.img_label.config(image=empty_sprite)
        for lbl in self.stats_labels: lbl.config(text="")
        self.dropdown.withdraw()
        self.entry.focus_set()

    def update_preview(self, name):
        p = sprite_dict.get(name)
        if p:
            img = Image.open(p).resize(SPRITE_SIZE)
            tkimg = ImageTk.PhotoImage(img)
            self.img_label.config(image=tkimg)
            self.img_label.image = tkimg
        else:
            self.img_label.config(image=empty_sprite)

        row = get_pokemon_row(name)
        for lbl, stat in zip(self.stats_labels, selected_stats):
            col = STAT_COLS.get(stat)
            val = row.get(col, "—") if isinstance(row, pd.Series) and col else "—"
            lbl.config(text=fmt_stat(val))

    def _on_hover(self, event):
        idx = self.listbox.nearest(event.y)
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)

    def _click_outside(self, event):
        widgets = [self.entry, self.arrow_btn, self.clear_btn, self.dropdown, self.listbox]
        if all(event.widget != w and not str(event.widget).startswith(str(w)) for w in widgets):
            self.dropdown.withdraw()

opponent_frame = ttk.LabelFrame(main, text="Opponent Team")
opponent_frame.pack(side="left", anchor="nw", padx=10)  # no stretch

opponent_entries = []
for i in range(NUM_POKEMON):
    img_lbl = ttk.Label(opponent_frame, image=empty_sprite)
    img_lbl.grid(row=i, column=0, padx=2, pady=4, sticky="w")

    stats_frame = ttk.Frame(opponent_frame)
    stats_frame.grid(row=i, column=2, padx=4, pady=4, sticky="nw")
    for c, ms in COL_SIZES.items():
        stats_frame.grid_columnconfigure(c, weight=0, minsize=ms)

    entry = AutocompleteEntry(opponent_frame, all_pokemon_names, img_lbl, stats_frame)
    entry.grid(row=i, column=1, padx=6, pady=4, sticky="w")
    opponent_entries.append(entry)

bottom = ttk.Frame(root)
bottom.pack(side="top", anchor="w", padx=10, pady=(0,10))

exclude_legendaries_var = tk.BooleanVar(value=False)
tk.Checkbutton(bottom, text="Exclude Legendaries (Friendly)", variable=exclude_legendaries_var).pack(side="left")

# loading gif, broken rn
loading_label = tk.Label(bottom)
loading_label.pack(side="right", padx=8)

loading_frames, spinner_job, loading_running, loading_index = [], None, False, 0
try:
    gif = Image.open(resource_path("loading.gif"))
    for f in range(gif.n_frames):
        gif.seek(f)
        loading_frames.append(ImageTk.PhotoImage(gif.copy().resize((24, 24))))
except Exception as e:
    print("Failed to load loading GIF:", e)

def start_spinner():
    global loading_running, loading_index, spinner_job
    if not loading_frames: return
    loading_running = True; loading_index = 0
    loading_label.config(image=loading_frames[0])
    def step():
        global loading_index, spinner_job
        if not loading_running: spinner_job = None; return
        loading_index = (loading_index + 1) % len(loading_frames)
        loading_label.config(image=loading_frames[loading_index])
        spinner_job = root.after(80, step)
    step()

def stop_spinner():
    global loading_running, spinner_job
    loading_running = False
    if spinner_job is not None:
        root.after_cancel(spinner_job); spinner_job = None
    loading_label.config(image="")

# calls optimization script
def generate_team(opponent_team, exclude_legendaries=True):
    try:
        return optimize_team(opponent_team, include_legendaries=not exclude_legendaries, method="heuristic")
    except Exception as e:
        print("Error generating team:", e)
        return [{"name": "Error", "score": 0.0}] * NUM_POKEMON

def generate_and_display_team_thread():
    opponent_team = [e.var.get() for e in opponent_entries if e.var.get()]
    if not opponent_team: return
    start_spinner()

    def task():
        team = generate_team(opponent_team, exclude_legendaries=exclude_legendaries_var.get())

        def update_ui():
            stop_spinner()
            for i, info in enumerate(team):
                row = friendly_rows[i]
                name, score = info["name"], info["score"]
                row["name"].config(text=name)
                p = sprite_dict.get(name)
                if p:
                    im = Image.open(p).resize(SPRITE_SIZE)
                    tkim = ImageTk.PhotoImage(im)
                    row["sprite"].config(image=tkim); row["sprite"].image = tkim
                else:
                    row["sprite"].config(image=empty_sprite)

                data = get_pokemon_row(name)
                for lbl, stat in zip(row["stats_labels"], friendly_stats):
                    if stat == "Score":
                        lbl.config(text=f"{score:.2f}")
                    else:
                        col = STAT_COLS.get(stat)
                        val = data.get(col, "—") if isinstance(data, pd.Series) and col else "—"
                        lbl.config(text=fmt_stat(val))

        root.after(0, update_ui)

    threading.Thread(target=task, daemon=True).start()

tk.Button(bottom, text="Generate Friendly Team", command=generate_and_display_team_thread).pack(side="left", padx=(10,0))

# fix window size
root.update_idletasks()
w, h = root.winfo_reqwidth(), root.winfo_reqheight()
root.minsize(w, h)
root.maxsize(w, h)
root.resizable(False, False)

root.mainloop()