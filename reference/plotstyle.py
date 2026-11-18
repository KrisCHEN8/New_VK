import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GROUP_COLOR = {"Envelope (outdoor)": PAL[0], "Solar": PAL[3], "Occupancy (CO2)": PAL[4], "Schedule (clock)": PAL[1],
               "Chilled beam": PAL[2], "Radiator": "#8b5cf6", "Building mass": PAL[5], "Constant gain": "#9a9890"}
def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURF)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]: ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8); ax.grid(color=GRID, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    if title: ax.set_title(title, loc="left", fontsize=10, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=8.5, color=INK2)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)
def fig(*a, **k):
    f, ax = plt.subplots(*a, facecolor=SURF, **k); return f, ax
