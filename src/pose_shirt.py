import torch
import trimesh
import numpy as np
import pickle
from pytorch3d.structures import Meshes
from pytorch3d.ops import knn_points

# --- CONFIGURATION ---
SMPL_FILE = "SMPL_NEUTRAL.pkl" 
#SHIRT_FILE = "shirt_robust_final.obj"
SHIRT_FILE = "shirt_reset_final.obj"
OUTPUT_FILE = "shirt_posed_CORRECTED.obj"
device = torch.device("cuda:0")

# --- 1. LOAD SMPL DATA ---
def load_smpl_data(path):
    print(f"Loading SMPL from {path}...")
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    
    # Standard Loading & Fixes
    J_regressor = torch.tensor(data['J_regressor'].toarray(), dtype=torch.float32, device=device)
    weights = torch.tensor(data['weights'], dtype=torch.float32, device=device)
    v_template = torch.tensor(data['v_template'], dtype=torch.float32, device=device)
    
    # Fix Integer Types
    faces_np = data['f'].astype(np.int32)
    faces = torch.tensor(faces_np, dtype=torch.int64, device=device)
    
    parents_np = data['kintree_table'][0].astype(np.int32)
    parents_np[0] = -1 # Safety Force
    parents = torch.tensor(parents_np, dtype=torch.int64, device=device)
    
    J_template = torch.matmul(J_regressor, v_template)
    
    return v_template, J_template, weights, parents, faces

# --- 2. LOAD SHIRT ---
def load_shirt(path):
    print(f"Loading Shirt from {path}...")
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int64, device=device)
    return verts, faces

# --- 3. LBS FUNCTION (The Engine) ---
def lbs(verts, joints, weights, pose_rotation, parents):
    N = verts.shape[0]
    num_joints = 24
    
    # Initial Matrices
    G = torch.eye(4, device=device).unsqueeze(0).repeat(num_joints, 1, 1)
    local_transforms = torch.eye(4, device=device).unsqueeze(0).repeat(num_joints, 1, 1)
    local_transforms[:, :3, :3] = pose_rotation
    local_transforms[:, :3, 3] = joints 
    
    tr_inv = torch.eye(4, device=device).unsqueeze(0).repeat(num_joints, 1, 1)
    tr_inv[:, :3, 3] = -joints
    
    # Forward Kinematics
    G_final = torch.zeros(num_joints, 4, 4, device=device)
    for i in range(num_joints):
        parent_idx = parents[i].item()
        if parent_idx == -1: 
            G_final[i] = local_transforms[i]
        else:
            G_final[i] = torch.matmul(G_final[parent_idx], local_transforms[i])
            
    G_skin = torch.matmul(G_final, tr_inv)
    
    # Apply Weights & Transform
    T = torch.tensordot(weights, G_skin, dims=([1], [0])) 
    ones = torch.ones(N, 1, device=device)
    verts_homo = torch.cat([verts, ones], dim=1).unsqueeze(2) 
    new_verts = torch.matmul(T, verts_homo)[:, :3, 0]
    
    return new_verts

# --- MAIN EXECUTION ---
print("=== SMART RIGGING: FIXING BIND POSE ===")

# A. Load Data
smpl_verts, smpl_joints, smpl_weights, parents, smpl_faces = load_smpl_data(SMPL_FILE)
shirt_verts, shirt_faces = load_shirt(SHIRT_FILE)

# --- STEP B: CREATE "GRAVITY POSE" SKELETON ---
# We must lower the skeleton's arms to match the shirt BEFORE binding.
print("1. Creating Gravity-Pose Skeleton (Arms Down)...")
gravity_rots = torch.eye(3, device=device).unsqueeze(0).repeat(24, 1, 1)

# Rotate Shoulders DOWN (70 degrees)
angle_down = np.pi / 2.5 # ~70 degrees down
rot_down = torch.tensor([
    [np.cos(angle_down), -np.sin(angle_down), 0],
    [np.sin(angle_down),  np.cos(angle_down), 0],
    [0, 0, 1]
], dtype=torch.float32, device=device)

gravity_rots[16] = rot_down # Left Shoulder
gravity_rots[17] = rot_down # Right Shoulder (Mirror approximation)

# Deform SMPL into Gravity Pose
# Note: We use SMPL weights on itself to pose it
smpl_gravity_verts = lbs(smpl_verts, smpl_joints, smpl_weights, gravity_rots, parents)

# --- STEP C: BIND SHIRT TO GRAVITY SKELETON ---
# Now the shirt sleeve is next to the Arm Bone, not the Hip Bone!
print("2. Binding Shirt to Gravity Skeleton...")
knn = knn_points(shirt_verts.unsqueeze(0), smpl_gravity_verts.unsqueeze(0), K=1)
nearest_idx = knn.idx.squeeze() 
shirt_weights = smpl_weights[nearest_idx] # Steal weights from the posed skeleton

# --- STEP D: POSE IT UP ---
# Now we apply the "Lift" pose to the shirt
print("3. Lifting Shirt Arms...")
lift_rots = torch.eye(3, device=device).unsqueeze(0).repeat(24, 1, 1)

# Rotate Shoulders UP (Target Pose)
# We use a smaller angle since we are starting from a neutral-ish binding
angle_up = -np.pi / 4 # 45 degrees up
rot_up = torch.tensor([
    [np.cos(angle_up), -np.sin(angle_up), 0],
    [np.sin(angle_up),  np.cos(angle_up), 0],
    [0, 0, 1]
], dtype=torch.float32, device=device)

lift_rots[16] = rot_up 
lift_rots[17] = rot_up 

# Animate!
posed_verts = lbs(shirt_verts, smpl_joints, shirt_weights, lift_rots, parents)

# E. Save
final_mesh = trimesh.Trimesh(vertices=posed_verts.cpu().numpy(), faces=shirt_faces.cpu().numpy())
final_mesh.export(OUTPUT_FILE)
print(f"DONE. Saved CORRECTED mesh to: {OUTPUT_FILE}")
