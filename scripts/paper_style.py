import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm

# Okabe–Ito palette (colorblind-safe) for categorical series
COLORS = {
    "blue":       "#0072B2",
    "orange":     "#E69F00",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "sky":        "#56B4E9",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
    "black":      "#000000",
}
PALETTE = [COLORS[k] for k in ["purple","blue","orange","green","sky","vermillion","yellow","black"]]

def make_seq_cmap(kind="viridis"):
    if kind == "magenta":  # optional: white -> light purple -> magenta
        return LinearSegmentedColormap.from_list(
            "paper_magenta", [(1,1,1), (0.97,0.96,1.0), (0.80,0.00,0.50)], N=256
        )
    return cm.get_cmap("viridis")  # default sequential cmap

CMAP_SEQ = make_seq_cmap("viridis")

def make_div_cmap():
    try:
        import cmcrameri.cm as cmc  # if installed
        return cmc.vik
    except Exception:
        return cm.get_cmap("coolwarm")

CMAP_DIV = make_div_cmap()

def set_theme():
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.prop_cycle": cycler(color=PALETTE),
        "axes.linewidth": 0.8,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    return CMAP_SEQ, CMAP_DIV, COLORS
