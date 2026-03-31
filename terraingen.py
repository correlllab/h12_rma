"""
terraingen.py -- Headless Isaac Gym terrain visualizer.

No display required. Renders via Isaac Gym camera sensors and saves PNG images:
  terrain_overview.png       -- isometric view of the full grid
  terrain_topdown.png        -- top-down view of the full grid
  terrain_col_N_<name>.png   -- close-up of each terrain type (all difficulties)

Terrain layout (curriculum grid):
  rows (x-axis) = difficulty 0 (easy) -> 1 (hard)
  cols (y-axis) = terrain type

  Col 0  smooth slope
  Col 1  rough slope
  Col 2  stairs down
  Col 3  stairs up
  Col 4  discrete obstacles
  Col 5  stepping stones
  Col 6  gaps

Usage:
    python terraingen.py
    python terraingen.py --mesh heightfield
    python terraingen.py --rows 5 --cols 7
    python terraingen.py --out /tmp/terrain_imgs
"""

# isaacgym must come before any torch import
import isaacgym
from isaacgym import gymapi, terrain_utils

import numpy as np
import sys
import os

try:
    from PIL import Image
    def save_img(arr, path):
        Image.fromarray(arr).save(path)
except ImportError:
    import matplotlib.pyplot as plt
    def save_img(arr, path):
        plt.imsave(path, arr)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TerrainCfg:
    terrain_proportions = [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]
    mesh_type        = 'trimesh'
    horizontal_scale = 0.1
    vertical_scale   = 0.005
    slope_treshold   = 0.75
    terrain_length   = 8.0
    terrain_width    = 8.0
    num_rows         = 10
    num_cols         = 7
    border_size      = 20.
    curriculum       = True
    static_friction  = 1.0
    dynamic_friction = 1.0
    restitution      = 0.0


TERRAIN_NAMES = [
    "smooth_slope",
    "rough_slope",
    "stairs_down",
    "stairs_up",
    "discrete_obstacles",
    "stepping_stones",
    "gaps",
]


# ---------------------------------------------------------------------------
# Terrain generation (no legged_gym imports)
# ---------------------------------------------------------------------------
def _gap_terrain(t, gap_size, platform_size=1.):
    gap_size      = int(gap_size      / t.horizontal_scale)
    platform_size = int(platform_size / t.horizontal_scale)
    cx, cy = t.length // 2, t.width // 2
    x1 = (t.length - platform_size) // 2;  x2 = x1 + gap_size
    y1 = (t.width  - platform_size) // 2;  y2 = y1 + gap_size
    t.height_field_raw[cx-x2:cx+x2, cy-y2:cy+y2] = -1000
    t.height_field_raw[cx-x1:cx+x1, cy-y1:cy+y1] = 0


def _pit_terrain(t, depth, platform_size=1.):
    depth         = int(depth         / t.vertical_scale)
    platform_size = int(platform_size / t.horizontal_scale / 2)
    x1 = t.length // 2 - platform_size;  x2 = t.length // 2 + platform_size
    y1 = t.width  // 2 - platform_size;  y2 = t.width  // 2 + platform_size
    t.height_field_raw[x1:x2, y1:y2] = -depth


class Terrain:
    def __init__(self, cfg):
        self.cfg = cfg
        self.env_length = cfg.terrain_length
        self.env_width  = cfg.terrain_width
        self.proportions = [
            np.sum(cfg.terrain_proportions[:i+1])
            for i in range(len(cfg.terrain_proportions))
        ]
        self.width_per_env_pixels  = int(cfg.terrain_width  / cfg.horizontal_scale)
        self.length_per_env_pixels = int(cfg.terrain_length / cfg.horizontal_scale)
        self.border   = int(cfg.border_size / cfg.horizontal_scale)
        self.tot_cols = cfg.num_cols * self.width_per_env_pixels  + 2 * self.border
        self.tot_rows = cfg.num_rows * self.length_per_env_pixels + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        if cfg.curriculum:
            self._curriculum()
        else:
            self._randomized()

        self.heightsamples = self.height_field_raw
        if cfg.mesh_type == 'trimesh':
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw, cfg.horizontal_scale,
                cfg.vertical_scale, cfg.slope_treshold)

    def _curriculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                self._add(self._make(j / self.cfg.num_cols + 0.001,
                                     i / self.cfg.num_rows), i, j)

    def _randomized(self):
        for k in range(self.cfg.num_rows * self.cfg.num_cols):
            i, j = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            self._add(self._make(np.random.uniform(0, 1),
                                 np.random.choice([0.5, 0.75, 0.9])), i, j)

    def _make(self, choice, difficulty):
        cfg = self.cfg
        s = terrain_utils.SubTerrain("terrain",
            width=self.width_per_env_pixels, length=self.width_per_env_pixels,
            vertical_scale=cfg.vertical_scale, horizontal_scale=cfg.horizontal_scale)
        slope   = difficulty * 0.4
        step_h  = 0.05 + 0.18 * difficulty
        obs_h   = 0.05 + difficulty * 0.2
        st_size = 1.5 * (1.05 - difficulty)
        st_dist = 0.05 if difficulty == 0 else 0.1
        gap_sz  = 1.0 * difficulty
        p = self.proportions
        if   choice < p[0]:
            if choice < p[0]/2: slope *= -1
            terrain_utils.pyramid_sloped_terrain(s, slope=slope, platform_size=3.)
        elif choice < p[1]:
            terrain_utils.pyramid_sloped_terrain(s, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(s, -0.05, 0.05, 0.005, 0.2)
        elif choice < p[3]:
            if choice < p[2]: step_h *= -1
            terrain_utils.pyramid_stairs_terrain(s, step_width=0.31,
                                                 step_height=step_h, platform_size=3.)
        elif choice < p[4]:
            terrain_utils.discrete_obstacles_terrain(s, obs_h, 1., 2., 20, platform_size=3.)
        elif choice < p[5]:
            terrain_utils.stepping_stones_terrain(s, stone_size=st_size,
                                                  stone_distance=st_dist,
                                                  max_height=0., platform_size=4.)
        elif choice < p[6]:
            _gap_terrain(s, gap_size=gap_sz, platform_size=3.)
        else:
            _pit_terrain(s, depth=gap_sz, platform_size=4.)
        return s

    def _add(self, sub, row, col):
        sx = self.border + row * self.length_per_env_pixels
        sy = self.border + col * self.width_per_env_pixels
        self.height_field_raw[sx:sx+self.length_per_env_pixels,
                               sy:sy+self.width_per_env_pixels] = sub.height_field_raw
        x1 = int((self.env_length/2.-1) / sub.horizontal_scale)
        x2 = int((self.env_length/2.+1) / sub.horizontal_scale)
        y1 = int((self.env_width /2.-1) / sub.horizontal_scale)
        y2 = int((self.env_width /2.+1) / sub.horizontal_scale)
        oz = np.max(sub.height_field_raw[x1:x2, y1:y2]) * sub.vertical_scale
        self.env_origins[row, col] = [(row+0.5)*self.env_length,
                                      (col+0.5)*self.env_width, oz]


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def terrain_stats(terrain, cfg):
    """Print per-column height statistics."""
    hf  = terrain.height_field_raw.astype(np.float32) * cfg.vertical_scale
    b   = terrain.border
    lpx = terrain.length_per_env_pixels
    wpx = terrain.width_per_env_pixels
    nr, nc = cfg.num_rows, cfg.num_cols

    print("\n  Height statistics per terrain type (metres)")
    print(f"  {'Col':<4} {'Type':<22} {'rows':<6} {'min':>7} {'max':>7} {'mean':>7}")
    print(f"  {'':-<4} {'':-<22} {'':-<6} {'':-<7} {'':-<7} {'':-<7}")
    for j in range(nc):
        sy = b + j * wpx
        col_vals = []
        for i in range(nr):
            sx = b + i * lpx
            patch = hf[sx:sx+lpx, sy:sy+wpx]
            col_vals.append(patch)
        v = np.concatenate([p.flatten() for p in col_vals])
        name = TERRAIN_NAMES[j] if j < len(TERRAIN_NAMES) else f"type_{j}"
        print(f"  {j:<4} {name:<22} {nr:<6} {v.min():>7.3f} {v.max():>7.3f} {v.mean():>7.3f}")

    flat = hf[b:b+nr*lpx, b:b+nc*wpx]
    print(f"\n  Overall grid  min={flat.min():.3f}  max={flat.max():.3f}  "
          f"mean={flat.mean():.3f}")


# ---------------------------------------------------------------------------
# Isaac Gym setup
# ---------------------------------------------------------------------------
def create_sim():
    gym = gymapi.acquire_gym()
    p = gymapi.SimParams()
    p.dt = 0.02;  p.substeps = 1
    p.up_axis = gymapi.UP_AXIS_Z
    p.gravity  = gymapi.Vec3(0., 0., -9.81)
    p.physx.solver_type             = 1
    p.physx.num_position_iterations = 4
    p.physx.num_velocity_iterations = 0
    p.physx.num_threads             = 4
    p.physx.use_gpu                 = True
    p.use_gpu_pipeline              = False
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, p)
    if sim is None:
        raise RuntimeError("gym.create_sim() failed")
    return gym, sim


def add_terrain(gym, sim, terrain, cfg):
    offset = -cfg.border_size
    if cfg.mesh_type == 'trimesh':
        p = gymapi.TriangleMeshParams()
        p.nb_vertices  = terrain.vertices.shape[0]
        p.nb_triangles = terrain.triangles.shape[0]
        p.transform.p.x = offset;  p.transform.p.y = offset;  p.transform.p.z = 0.
        p.static_friction  = cfg.static_friction
        p.dynamic_friction = cfg.dynamic_friction
        p.restitution      = cfg.restitution
        gym.add_triangle_mesh(sim, terrain.vertices.flatten(order='C'),
                              terrain.triangles.flatten(order='C'), p)
    else:
        p = gymapi.HeightFieldParams()
        p.column_scale = cfg.horizontal_scale;  p.row_scale = cfg.horizontal_scale
        p.vertical_scale = cfg.vertical_scale
        p.nbColumns = terrain.tot_cols;  p.nbRows = terrain.tot_rows
        p.transform.p.x = offset;  p.transform.p.y = offset;  p.transform.p.z = 0.
        p.static_friction  = cfg.static_friction
        p.dynamic_friction = cfg.dynamic_friction
        p.restitution      = cfg.restitution
        gym.add_heightfield(sim, terrain.heightsamples.flatten(order='C'), p)


def make_env(gym, sim):
    """Minimal single environment — needed to attach camera sensors."""
    lo = gymapi.Vec3(-1., -1., -1.)
    hi = gymapi.Vec3( 1.,  1.,  1.)
    return gym.create_env(sim, lo, hi, 1)


def make_camera(gym, sim, env, width, height):
    props = gymapi.CameraProperties()
    props.width  = width
    props.height = height
    props.enable_tensors = False
    return gym.create_camera_sensor(env, props)


def capture(gym, sim, env, cam):
    """Render and return RGBA uint8 array (H, W, 4)."""
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)
    img = gym.get_camera_image(sim, env, cam, gymapi.IMAGE_COLOR)
    return img


# ---------------------------------------------------------------------------
# Rendering shots
# ---------------------------------------------------------------------------
def render_all(gym, sim, env, cfg, out_dir):
    gx = cfg.num_rows * cfg.terrain_length   # full grid x extent
    gy = cfg.num_cols * cfg.terrain_width    # full grid y extent
    cx, cy = gx / 2., gy / 2.

    W, H = 1920, 1080
    cam = make_camera(gym, sim, env, W, H)

    # warm up the sim so terrain is fully loaded before rendering
    for _ in range(5):
        gym.simulate(sim)
        gym.fetch_results(sim, True)

    shots = []

    # --- isometric overview ------------------------------------------------
    gym.set_camera_location(cam, env,
        gymapi.Vec3(cx - gx*0.5, -gy*0.15, max(gx, gy)*0.75),
        gymapi.Vec3(cx, cy, 0.))
    img = capture(gym, sim, env, cam)
    path = os.path.join(out_dir, "terrain_overview.png")
    save_img(img, path)
    shots.append(("overview (isometric)", path))

    # --- top-down ----------------------------------------------------------
    gym.set_camera_location(cam, env,
        gymapi.Vec3(cx, cy, max(gx, gy)),
        gymapi.Vec3(cx, cy + 0.001, 0.))
    img = capture(gym, sim, env, cam)
    path = os.path.join(out_dir, "terrain_topdown.png")
    save_img(img, path)
    shots.append(("top-down", path))

    # --- per-column (one per terrain type) ---------------------------------
    n = min(cfg.num_cols, len(TERRAIN_NAMES))
    for j in range(n):
        col_cx = (j + 0.5) * cfg.terrain_width    # world y centre of column j
        col_mid_x = gx / 2.

        # Camera off to the side, looking across all difficulty rows
        gym.set_camera_location(cam, env,
            gymapi.Vec3(-8., col_cx, 25.),
            gymapi.Vec3(col_mid_x, col_cx, 1.))
        img = capture(gym, sim, env, cam)
        name = TERRAIN_NAMES[j]
        path = os.path.join(out_dir, f"terrain_col{j:02d}_{name}.png")
        save_img(img, path)
        shots.append((f"col {j} – {name}", path))

    return shots


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    cfg  = TerrainCfg()
    out  = "terrain_renders"
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok == '--mesh'  and i+1 < len(argv): cfg.mesh_type = argv[i+1];       i += 2
        elif tok == '--rows' and i+1 < len(argv): cfg.num_rows  = int(argv[i+1]); i += 2
        elif tok == '--cols' and i+1 < len(argv): cfg.num_cols  = int(argv[i+1]); i += 2
        elif tok == '--out'  and i+1 < len(argv): out           = argv[i+1];      i += 2
        else: i += 1
    return cfg, out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg, out_dir = parse_args()
    os.makedirs(out_dir, exist_ok=True)

    # ---- build terrain ----------------------------------------------------
    print("Building terrain ...", flush=True)
    terrain = Terrain(cfg)
    if cfg.mesh_type == 'trimesh':
        print(f"  vertices={terrain.vertices.shape[0]}  "
              f"triangles={terrain.triangles.shape[0]}")
    else:
        print(f"  heightfield={terrain.height_field_raw.shape}")

    print(f"\nGrid layout: {cfg.num_rows} rows (difficulty) "
          f"x {cfg.num_cols} cols (terrain type)  "
          f"patch={cfg.terrain_length}m x {cfg.terrain_width}m  mesh={cfg.mesh_type}")
    print("  Row 0 = easiest, last row = hardest\n")
    n = min(cfg.num_cols, len(TERRAIN_NAMES))
    print("  Col | Type")
    print("  ----|--------------------")
    for j in range(n):
        print(f"   {j:2d} | {TERRAIN_NAMES[j]}")

    terrain_stats(terrain, cfg)

    # ---- Isaac Gym --------------------------------------------------------
    print("\nStarting Isaac Gym (headless) ...", flush=True)
    gym, sim = create_sim()
    add_terrain(gym, sim, terrain, cfg)
    env = make_env(gym, sim)

    # ---- render & save ----------------------------------------------------
    print(f"Rendering {2 + min(cfg.num_cols, len(TERRAIN_NAMES))} shots ...", flush=True)
    shots = render_all(gym, sim, env, cfg, out_dir)

    print(f"\nSaved to: {os.path.abspath(out_dir)}/")
    for label, path in shots:
        size_kb = os.path.getsize(path) // 1024
        print(f"  {os.path.basename(path):<45}  ({size_kb} KB)  [{label}]")

    gym.destroy_sim(sim)
    print("\nDone.")


if __name__ == '__main__':
    main()
