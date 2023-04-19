import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import shutil

from utils.human_models.ObjFile import ObjFile

os.environ["PYOPENGL_PLATFORM"] = "egl"

def vis_keypoints_with_skeleton(img, kps, kps_lines, kp_thresh=0.4, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_keypoints(img, kps, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=3, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_keypoints_pcf(img, kps_p, kps_c, kps_f, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    colors_p = (255,0,0)
    colors_c = (0,255,0)
    colors_f = (0,0,255)

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps_p)):
        p = kps_p[i][0].astype(np.int32), kps_p[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=3, color=colors_p, thickness=-1, lineType=cv2.LINE_AA)

        p = kps_c[i][0].astype(np.int32), kps_c[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=3, color=colors_c, thickness=-1, lineType=cv2.LINE_AA)

        p = kps_f[i][0].astype(np.int32), kps_f[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=3, color=colors_f, thickness=-1, lineType=cv2.LINE_AA)
        
    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

def vis_mesh(img, mesh_vertex, alpha=0.5):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(mesh_vertex))]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    mask = np.copy(img)

    # Draw the mesh
    for i in range(len(mesh_vertex)):
        p = mesh_vertex[i][0].astype(np.int32), mesh_vertex[i][1].astype(np.int32)
        cv2.circle(mask, p, radius=1, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, mask, alpha, 0)

def vis_3d_skeleton(kpt_3d, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [np.array((c[2], c[1], c[0])) for c in colors]

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d[i1,0], kpt_3d[i2,0]])
        y = np.array([kpt_3d[i1,1], kpt_3d[i2,1]])
        z = np.array([kpt_3d[i1,2], kpt_3d[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors[l], linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d[i1,0], kpt_3d[i1,2], -kpt_3d[i1,1], c=colors[l], marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d[i2,0], kpt_3d[i2,2], -kpt_3d[i2,1], c=colors[l], marker='o')

    # x_r = np.array([0, cfg.input_shape[1]], dtype=np.float32)
    # y_r = np.array([0, cfg.input_shape[0]], dtype=np.float32)
    # z_r = np.array([0, 1], dtype=np.float32)
    
    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')

    ax.set_xlim([-0.1,0.1])
    ax.set_ylim([-0.1,0.1])
    ax.set_zlim([-0.1,0.1])

    ax.legend()
    
    plt.savefig(filename)
    # plt.show()
    # cv2.waitKey(0)

def vis_3d_skeleton_pcf(kpt_3d_p, kpt_3d_c, kpt_3d_f, kpt_3d_vis, kps_lines, filename=None):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    colors_p = (1,0,0)
    colors_c = (0,1,0)
    colors_f = (0,0,1)

    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d_p[i1,0], kpt_3d_p[i2,0]])
        y = np.array([kpt_3d_p[i1,1], kpt_3d_p[i2,1]])
        z = np.array([kpt_3d_p[i1,2], kpt_3d_p[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors_p, linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d_p[i1,0], kpt_3d_p[i1,2], -kpt_3d_p[i1,1], c=colors_p, marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d_p[i2,0], kpt_3d_p[i2,2], -kpt_3d_p[i2,1], c=colors_p, marker='o')

        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d_c[i1,0], kpt_3d_c[i2,0]])
        y = np.array([kpt_3d_c[i1,1], kpt_3d_c[i2,1]])
        z = np.array([kpt_3d_c[i1,2], kpt_3d_c[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors_c, linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d_c[i1,0], kpt_3d_c[i1,2], -kpt_3d_c[i1,1], c=colors_c, marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d_c[i2,0], kpt_3d_c[i2,2], -kpt_3d_c[i2,1], c=colors_c, marker='o')


        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        x = np.array([kpt_3d_f[i1,0], kpt_3d_f[i2,0]])
        y = np.array([kpt_3d_f[i1,1], kpt_3d_f[i2,1]])
        z = np.array([kpt_3d_f[i1,2], kpt_3d_f[i2,2]])

        if kpt_3d_vis[i1,0] > 0 and kpt_3d_vis[i2,0] > 0:
            ax.plot(x, z, -y, c=colors_f, linewidth=2)
        if kpt_3d_vis[i1,0] > 0:
            ax.scatter(kpt_3d_f[i1,0], kpt_3d_f[i1,2], -kpt_3d_f[i1,1], c=colors_f, marker='o')
        if kpt_3d_vis[i2,0] > 0:
            ax.scatter(kpt_3d_f[i2,0], kpt_3d_f[i2,2], -kpt_3d_f[i2,1], c=colors_f, marker='o')


    # x_r = np.array([0, cfg.input_shape[1]], dtype=np.float32)
    # y_r = np.array([0, cfg.input_shape[0]], dtype=np.float32)
    # z_r = np.array([0, 1], dtype=np.float32)
    
    if filename is None:
        ax.set_title('3D vis')
    else:
        ax.set_title(filename)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Z Label')
    ax.set_zlabel('Y Label')

    ax.set_xlim([-0.1,0.1])
    ax.set_ylim([-0.1,0.1])
    ax.set_zlim([-0.1,0.1])

    ax.legend()
    
    plt.savefig(filename)
    # plt.show()
    # cv2.waitKey(0)

def save_obj(v, f, file_name='output.obj'):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()


def seq2video(mesh_p, mesh_c, mesh_f, blur_img, mano_face, save_name, n_interpolate=2, rm_tmp=True, fps=5):
    meshes = [mesh_p, mesh_c, mesh_f]
    
    # interpolation for smooth motion
    for _ in range(n_interpolate):
        tmp_meshes = []
        for idx in range(len(meshes)):
            tmp_meshes.append(meshes[idx])
            if (idx + 1) < len(meshes):
                tmp_meshes.append((meshes[idx]+meshes[idx+1])/2.)
        meshes = tmp_meshes
    
    # save obj and convert to img
    imgs = []
    tmp_dir, basename = osp.split(save_name)
    basename, _ = osp.splitext(basename)
    tmp_dir = osp.join(tmp_dir, f'{basename}_tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    for idx, mesh in enumerate(meshes):
        save_obj(mesh*np.array([1,-1,-1]), mano_face, osp.join(tmp_dir, '{:02d}.obj'.format(idx)))
        ob = ObjFile(osp.join(tmp_dir, '{:02d}.obj'.format(idx)))
        ob.Plot(osp.join(tmp_dir, '{:02d}.png'.format(idx)), elevation=90, azim=-90, dpi=None, scale=None, animate=None)
        imgs.append(cv2.imread(osp.join(tmp_dir, '{:02d}.png'.format(idx))))
    if rm_tmp:
        shutil.rmtree(tmp_dir)
        
    # make video
    frames = []
    for img in imgs:
        height, width, _ = img.shape
        blur_img = cv2.resize(blur_img, (width, height))
        cat_img = np.concatenate((blur_img, img),axis = 1)
        height, width, _ = cat_img.shape
        size = (width, height)
        frames.append(cat_img)
    
    out = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for frame in frames:
        out.write(frame)
        
    return out.release()
