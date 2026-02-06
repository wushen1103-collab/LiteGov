# Load protein and ligands
load docking/receptors/6LU7/receptor.pdbqt, prot
load docking/genmol-outs/6LU7/raw/raw_tau1.2_seed1_0031_out.pdbqt, raw
load docking/genmol-outs/6LU7/lite/lite_tau0.8_seed2_0098_out.pdbqt, lite

remove solvent
remove resn HOH
remove hydro

# ----- pocket-only protein (cartoon only) -----
# residues within 7 ? of either ligand
select pocket_res, byres (prot within 8 of raw or prot within 8 of lite)
create prot_pocket, pocket_res

# hide full protein
hide everything, all
hide everything, prot

# cartoon for pocket
show cartoon, prot_pocket
color grey90, prot_pocket
set cartoon_transparency, 0.7
set cartoon_smooth_loops, on
set cartoon_smooth_tube, on

# ----- ligands -----
show sticks, raw
show sticks, lite
color gray60, raw
color forest, lite
util.cnc raw
util.cnc lite
set stick_radius, 0.28

# ----- view -----
orient raw or lite
zoom raw or lite, 4.5

# ----- background & lighting -----
bg_color white
set ray_opaque_background, on
set ambient, 0.45
set direct, 0.60
set reflect, 0.02
set shininess, 20
set ray_trace_gain, 0.05
set ray_trace_mode, 0
set ray_shadow, off
set antialias, 2

# ----- render -----
ray 2400, 1800
png figs/case_6LU7_pose_with_pocket_clean.png, dpi=500
