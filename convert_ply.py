import modal

app = modal.App("convert-ply")
vol = modal.Volume.from_name("gaussian-outputs")

@app.function(
    image=modal.Image.debian_slim().pip_install("torch", "gsplat", "numpy", "plyfile"),
    volumes={"/outputs": vol},
    gpu="any"
)
def convert():
    import torch
    import numpy as np
    from plyfile import PlyData, PlyElement

    ck = torch.load("/outputs/driving_test/gsplat/ckpts/ckpt_6999_rank0.pt", map_location="cpu")
    splats = ck["splats"]

    means  = splats["means"].numpy()
    scales = splats["scales"].numpy()
    quats  = splats["quats"].numpy()
    opacs  = splats["opacities"].numpy()
    sh0    = splats["sh0"].numpy()

    if sh0.ndim == 3:
        sh0 = sh0[:, 0, :]

    N = means.shape[0]
    vertex = np.zeros(N, dtype=[
        ('x','f4'), ('y','f4'), ('z','f4'),
        ('scale_0','f4'), ('scale_1','f4'), ('scale_2','f4'),
        ('rot_0','f4'), ('rot_1','f4'), ('rot_2','f4'), ('rot_3','f4'),
        ('f_dc_0','f4'), ('f_dc_1','f4'), ('f_dc_2','f4'),
        ('opacity','f4'),
    ])

    vertex['x'], vertex['y'], vertex['z'] = means[:,0], means[:,1], means[:,2]
    vertex['scale_0'] = scales[:,0]
    vertex['scale_1'] = scales[:,1]
    vertex['scale_2'] = scales[:,2]
    vertex['rot_0'] = quats[:,0]
    vertex['rot_1'] = quats[:,1]
    vertex['rot_2'] = quats[:,2]
    vertex['rot_3'] = quats[:,3]
    vertex['f_dc_0'] = sh0[:,0]
    vertex['f_dc_1'] = sh0[:,1]
    vertex['f_dc_2'] = sh0[:,2]
    vertex['opacity'] = opacs

    out = "/outputs/driving_test/driving_test_full.ply"
    PlyData([PlyElement.describe(vertex, 'vertex')]).write(out)
    print(f"saved {N} gaussians -> {out}")
    vol.commit()

@app.local_entrypoint()
def main():
    convert.remote()
