import torch
import trimesh
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance, mesh_laplacian_smoothing

# --- CONFIG ---
# INPUT MUST BE THE ALIGNED MESH FROM STEP 1
SOURCE_PATH = "alignment_check.obj" 
SCAN_PATH = "scan_voxelated.obj"
OUTPUT_PATH = "shirt_reset_final.obj"
DEVICE = torch.device("cuda:0")

def load_mesh(path):
    print(f"Loading {path}...")
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=DEVICE)
    faces = torch.tensor(mesh.faces, dtype=torch.int64, device=DEVICE)
    return Meshes(verts=[verts], faces=[faces])

print("=== STEP 2: VOLUME-PRESERVING WARP (SOFT SHELL) ===")

# 1. Load Data
src_mesh = load_mesh(SOURCE_PATH)
tgt_mesh = load_mesh(SCAN_PATH)

# 2. Setup Optimizer
deform_verts = src_mesh.verts_packed().clone().detach().requires_grad_(True)
optimizer = torch.optim.Adam([deform_verts], lr=0.005)

# 3. Capture "Resting State" (The Anti-Collapse Force)
# We calculate the CURRENT length of every edge in the aligned shirt.
# We will force the code to respect these lengths so it doesn't shrink to zero.
with torch.no_grad():
    edges = src_mesh.edges_packed()
    verts = src_mesh.verts_packed()
    v0 = verts[edges[:, 0]]
    v1 = verts[edges[:, 1]]
    # Store the original lengths
    original_lengths = torch.norm(v0 - v1, dim=1)
    print(f"Locked in structural constraints for {len(original_lengths)} edges.")

# 4. The Loop
for i in range(501):
    optimizer.zero_grad()
    new_src_mesh = src_mesh.update_padded(deform_verts.unsqueeze(0))
    
    # --- LOSSES ---
    
    # A. Chamfer (The Pull) - Fits shirt to scan
    loss_chamfer, _ = chamfer_distance(new_src_mesh.verts_padded(), tgt_mesh.verts_padded())
    
    # B. Structure Preservation (The Anti-Collapse)
    # Get current lengths
    curr_edges = new_src_mesh.edges_packed()
    curr_verts = new_src_mesh.verts_packed()
    c0 = curr_verts[curr_edges[:, 0]]
    c1 = curr_verts[curr_edges[:, 1]]
    curr_lengths = torch.norm(c0 - c1, dim=1)
    
    # Penalty: (Current Length - Original Length)^2
    # This acts like a spring. If you stretch OR shrink it, penalty goes up.
    loss_preserve = torch.mean((curr_lengths - original_lengths) ** 2)
    
    # C. Smoothness (Cleanup)
    loss_smooth = mesh_laplacian_smoothing(new_src_mesh, method="uniform")
    
    # --- WEIGHTS ---
    # w_preserve = 500.0 is the "Iron Shield". It forbids collapsing.
    w_chamfer = 2.0
    w_preserve = 500.0 
    w_smooth = 20.0
    
    loss = (loss_chamfer * w_chamfer) + \
           (loss_preserve * w_preserve) + \
           (loss_smooth * w_smooth)
    
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print(f"[Step {i}] Fit Loss: {loss_chamfer.item():.4f} | Structure Loss: {loss_preserve.item():.6f}")

# 5. Save
final_verts = deform_verts.detach().cpu().numpy()
final_faces = src_mesh.faces_packed().detach().cpu().numpy()
trimesh.Trimesh(vertices=final_verts, faces=final_faces).export(OUTPUT_PATH)

print(f"DONE. Saved {OUTPUT_PATH}")
print("--> Download this file. This is your final geometry.")
