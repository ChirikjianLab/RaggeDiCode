import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from copy import deepcopy
import tqdm
from sklearn.cluster import AgglomerativeClustering
import open3d as o3d
from scipy.spatial import cKDTree
import trimesh

def remove_background(image): 
    # This function remove the background of the depth images, it removes all point that
    # has the same depth value with the 1st and 2nd depth value
    w, h = image.shape
    ground_value = np.max(image)
    image[image == ground_value] = 0
    floor_value = np.max(image)
    image[image == floor_value] = 0

    return image

def min_max_scale(img, mask, lim=(55,200)): # object mask is a 56*56 bool matrix
   img = img.astype(np.float32)
   img_min = np.min(img[mask])
   img_max = np.max(img[mask])
   img[mask] = np.clip((img[mask]-img_min)/(img_max-img_min+1e-8)*(lim[1]-lim[0])+lim[0], 0, 255)
   img[mask] = 255-img[mask]
   img[~mask] = 0
   
   return img.astype(np.uint8)

def std_scale(img, mask, lim=(55,200)): # object mask is a 56*56 bool matrix
   img = img.astype(np.float32)
   img_mean = np.mean(img[mask])
   img_std = np.std(img[mask])
   img[mask] = np.clip((img[mask]-img_mean)/(img_std*5+1e-8)*(lim[1]-lim[0])+255//2, 0, 255)
   img[mask] = 255-img[mask]
   img[~mask] = 0
   
   return img.astype(np.uint8)


def recenter(img,rgb): # use depth filtering only, no rgb info used 
    # import pdb; pdb.set_trace()
    w, h = img.shape
    N = 0
    cx = 0; cy = 0
    clothes_points = []
    for i in range(w):
        for j in range(h):
            is_white = (np.sum((rgb[i,j]>200).astype(np.int8)) == 3)
            # is_dark = (np.sum((rgb[i,j]<80).astype(np.int8)) == 3)
            if is_white or (np.abs(img[i,j].astype(np.float32)-205) < 5):
                img[i,j] = 255
            else:
                N += 1
                cx += i
                cy += j
                clothes_points.append([i, j])

    # import pdb; pdb.set_trace()
    cx = int(cx/N); cy = int(cy/N)
    clothes_points = np.asarray(clothes_points)
    x_min = np.min(clothes_points[:, 0])
    x_max = np.max(clothes_points[:, 0])
    y_min = np.min(clothes_points[:, 1])
    y_max = np.max(clothes_points[:, 1])
    d1 = x_max - x_min + 1
    d2 = y_max - y_min + 1
    if w/d1 < h/d2: 
        new_img = cv2.resize(img[x_min:x_max+1, y_min:y_max+1], (int(0.95*w*d2/d1), int(0.95*w)), interpolation=cv2.INTER_NEAREST)
        w_n, h_n = new_img.shape
        new_img = cv2.copyMakeBorder(new_img, int((w-w_n)/2), w-w_n-int((w-w_n)/2), int((h - h_n)/2), h-h_n-int((h - h_n)/2), borderType=cv2.BORDER_CONSTANT,value=255)
    else: 
        new_img = cv2.resize(img[x_min:x_max+1, y_min:y_max+1], (int(0.95*h), int(0.95*h*d1/d2)), interpolation=cv2.INTER_NEAREST)
        w_n, h_n = new_img.shape
        new_img = cv2.copyMakeBorder(new_img, int((w-w_n)/2), w-w_n-int((w-w_n)/2), int((h - h_n)/2), h-h_n-int((h - h_n)/2), borderType=cv2.BORDER_CONSTANT,value=255)

    return new_img

def find_mask(img): # return bool matrix
    return (img != 255)

def is_RGB(img):
    if np.sum( np.abs(img[:,:,0]-img[:,:,1]).reshape(-1,) ) != 0:
        return True
    return False


# TODO: change to std normalization, remember to clip to [-1, 1], renorm_scale = std * 4 + 1e-8
def process_pts(pts, renorm_scale=None): # rescale z axis and recenter
    # pts[:,0:2] /= 10
    pts = deepcopy(pts)
    cx = np.average(pts[:,0]); cy = np.average(pts[:,1]); cz = np.average(pts[:,2])
    if renorm_scale == None:
        renorm_scale = (np.max(pts[:,0])-np.min(pts[:,0]), \
                        np.max(pts[:,1])-np.min(pts[:,1]), \
                        (np.max(pts[:,1])-np.min(pts[:,1]))*0.4  )
        # offset = (cx, cy, cz)
    pts[:,0] = np.clip((pts[:,0]-cx)/renorm_scale[0]*2, -2, 2)
    pts[:,1] = np.clip((pts[:,1]-cy)/renorm_scale[1]*2, -2, 2)
    pts[:,2] = np.clip((pts[:,2]-cz)/renorm_scale[2]*2, -2, 2)
    # import pdb; pdb.set_trace()
    return pts, (cx, cy, cz), renorm_scale

def rev_process_pts(pts, offset, renorm_scale):
    pts[:,0] = (pts[:,0]*renorm_scale[0]/2)+offset[0]
    pts[:,1] = (pts[:,1]*renorm_scale[1]/2)+offset[1]
    pts[:,2] = (pts[:,2]*renorm_scale[2]/2)+offset[2]
    return pts

# TODO: [-2,2]->[0,255]: np.clip((img+1)*255/2, 0, 255)
def interval_to_uint(img):
    # import pdb; pdb.set_trace()
    img = np.clip((img+3)*255/6, 0, 255)
    return img.astype(np.uint8)

# TODO: [0,255]->[-1,1]: np.clip(img/255*2-1, -1, 1)
def uint_to_interval(img):
    img = img.astype(np.float16)
    return img/255*6-3

def show_output(depth_img, final_mp, current_pts, rev_pts):
    fig = plt.figure(figsize=(15,4))

    ax = fig.add_subplot(141)
    ax.imshow(depth_img)
    ax.set_title('Input Depth Image')

    ax = fig.add_subplot(142)
    ax.imshow(final_mp) # careful, rescale
    ax.set_title('Output Translation Map')

    ax = fig.add_subplot(143, projection='3d')
    ax.scatter(current_pts[:,0],current_pts[:,1],current_pts[:,2])
    ax.set_title('Ground Truth Point Cloud')
    
    ax = fig.add_subplot(144, projection='3d')
    ax.scatter(rev_pts[:,0],rev_pts[:,1],rev_pts[:,2])
    ax.set_title('Reversed Point Cloud')

    plt.show()

def pts_to_img(pts, ori_pts):
    mp = np.zeros((25, 25, 3))
    for x in range(25):
        for y in range(25):
            mp[24-y][x] = pts[x*25+y]-ori_pts[x*25+y]
    return mp

def img_to_pts(orig_pts, mp):
    pts = np.zeros_like(orig_pts)
    for x in range(25):
        for y in range(25):
            pts[x*25+y] = mp[24-y][x]+orig_pts[x*25+y]
    return pts

# Find the surface pts index
# Input: original pt cloud, grid size (for projection). Output: surface pts index
def find_surface_pts_ind(pts_orig, grid_size=30): 
    import heapq
    pts = deepcopy(pts_orig)
    pts[:,0] = (pts[:,0]-np.min(pts[:,0]))/(np.max(pts[:,0])-np.min(pts[:,0])) # shift to [0,1]
    pts[:,1] = (pts[:,1]-np.min(pts[:,1]))/(np.max(pts[:,1])-np.min(pts[:,1]))
    pts[:,0] = pts[:,0]*0.9+0.05 # shift to [0.05, 0.95], avoid singular boundary facts
    pts[:,1] = pts[:,1]*0.9+0.05
    # projected_grid = np.zeros((32, 32))
    grid_heap = [[[] for _ in range(grid_size+1)] for _ in range(grid_size+1)]
    for p in range(pts.shape[0]):
        x = int(pts[p, 0]*grid_size)
        y = int(pts[p, 1]*grid_size)
        heapq.heappush(grid_heap[x][y], (-pts[p,2], p))
    surface_pts_ind = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(grid_heap[i][j]) > 0:
                ind = heapq.heappop(grid_heap[i][j])[1]
                surface_pts_ind.append(ind)
    return surface_pts_ind

# Find surface points
# Input: original point cloud. Output: surface points
def surface_pts(pts):
    inds = find_surface_pts_ind(pts) # find index first
    surf_pts = np.zeros((len(inds), 3))
    for i in range(len(inds)):
        surf_pts[i,:] = pts[inds[i], :]
    return surf_pts


# From a depth image to point clouds (3D), (x, y, z) in [0, 1]
# Input: depth image, object mask (bool matrix). Output: (partial) point cloud of object 
def depth_to_pts(img, mask):
    img = img.astype(np.float32)
    w, h = img.shape
    pts = []
    for i in range(w):
        for j in range(h):
            if mask[i, j]:
                pts.append((j, h-i, 255-img[i,j]))
    pts = np.array(pts)
    scale = np.array([np.max(pts[:,0])-np.min(pts[:,0]),\
             np.max(pts[:,1])-np.min(pts[:,1]),\
             np.max(pts[:,2])-np.min(pts[:,2])])
    shift = np.array([np.min(pts[:,0]), np.min(pts[:,1]), np.min(pts[:,2])])
    pts = (pts - shift)/scale

    return pts, shift, scale

# find the SE(3) transform from the surface points to depth image point cloud
# return SE(3) transform (3,4) matrix; dim indicates whether flatten the point cloud or not for registration
def ICP_transform_mat(surf_pts, depth_pts, dim=2):  # currently ignore z direction matching
        if dim == 2:
            surf_pts[:, 2] = 0
            depth_pts[:, 2] = 0
        else:
            surf_pts[:, 2] /= (np.max(surf_pts[:, 2]) + 1e-7)
            depth_pts[:, 2] /= (np.max(depth_pts[:, 2]) + 1e-7)
        pcd_surf = o3d.geometry.PointCloud()
        pcd_surf.points = o3d.utility.Vector3dVector(surf_pts)

        pcd_depth = o3d.geometry.PointCloud()
        pcd_depth.points = o3d.utility.Vector3dVector(depth_pts)

        reg_p2p = o3d.pipelines.registration.registration_icp(pcd_surf, pcd_depth, 0.02, np.eye(4),\
                                                                o3d.pipelines.registration.TransformationEstimationPointToPoint())

        trans_mat = np.array(reg_p2p.transformation)
        return trans_mat
        
# transform the predicted pts in sim to the depth image space
def ICP_align(pred_pts, depth_pts, dim=2):
    surf_pts = surface_pts(pred_pts)
    trans_mat = ICP_transform_mat(surf_pts, depth_pts, dim)
    return np.hstack((surf_pts, np.ones((surf_pts.shape[0],1)))) @ trans_mat.T


def get_chamfer(pred_points, gt_points):
    pred_tree = cKDTree(pred_points)
    gt_tree = cKDTree(gt_points)
    forward_distance, _ = gt_tree.query(pred_points, k=1)
    backward_distance, _ = pred_tree.query(gt_points, k=1)
    forward_chamfer = np.mean(forward_distance)
    backward_chamfer = np.mean(backward_distance)
    symmetrical_chamfer = np.mean([forward_chamfer, backward_chamfer])
    return symmetrical_chamfer

def normalize_data(data):
    # nomalize to [0,1]
    ndata = (data - np.min(data)) / (np.max(data) - np.min(data))
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, max_d, min_d):
    ndata = (ndata + 1) / 2
    data = ndata * (max_d - min_d) + min_d
    return data


# output a list of triangular faces of the blanket (orignally grid)
def blanket_faces(): # single surface mesh
    mat = []
    for p in range(625):
        i = p // 25
        j = p % 25
        if i >= 24 or j >= 24:
            continue
        mat.append(np.array([p, p+1, p+25]))
        mat.append(np.array([p+1, p+26, p+25]))
    return np.array(mat, dtype=int)

# input full point cloud (625), output the curvature map
def cal_mean_curvature(pts, show=True):
    mesh = trimesh.Trimesh(vertices=pts, faces=blanket_faces())
    mean_curvature = trimesh.curvature.discrete_mean_curvature_measure(mesh, pts, radius=0.1)
    if show:
        mesh.show()
    curvature_map = np.zeros((25,25))
    for i in range(25):
        for j in range(25):
            curvature_map[i,j] = mean_curvature[i*25+j]
    return curvature_map



##################################
# input RGBD, output segmented depth image
def recenter_real(img,rgb): # use depth filtering only, no rgb info used 
    # import pdb; pdb.set_trace()
    w, h = img.shape
    N = 0
    cx = 0; cy = 0
    clothes_points = []
    for i in range(w):
        for j in range(h):
            if img[i,j] < 255: 
                N += 1
                cx += i
                cy += j
                clothes_points.append([i, j])                

    cx = int(cx/N); cy = int(cy/N)
    clothes_points = np.asarray(clothes_points)
    x_min = np.min(clothes_points[:, 0])
    x_max = np.max(clothes_points[:, 0])
    y_min = np.min(clothes_points[:, 1])
    y_max = np.max(clothes_points[:, 1])
    d1 = x_max - x_min + 1
    d2 = y_max - y_min + 1
    if w/d1 < h/d2: 
        new_img = cv2.resize(img[x_min:x_max+1, y_min:y_max+1], (int(0.95*w*d2/d1), int(0.95*w)), interpolation=cv2.INTER_NEAREST)
        w_n, h_n = new_img.shape
        new_img = cv2.copyMakeBorder(new_img, int((w-w_n)/2), w-w_n-int((w-w_n)/2), int((h - h_n)/2), h-h_n-int((h - h_n)/2), borderType=cv2.BORDER_CONSTANT,value=255)
    else: 
        new_img = cv2.resize(img[x_min:x_max+1, y_min:y_max+1], (int(0.95*h), int(0.95*h*d1/d2)), interpolation=cv2.INTER_NEAREST)
        w_n, h_n = new_img.shape
        new_img = cv2.copyMakeBorder(new_img, int((w-w_n)/2), w-w_n-int((w-w_n)/2), int((h - h_n)/2), h-h_n-int((h - h_n)/2), borderType=cv2.BORDER_CONSTANT,value=255)

    return new_img

def segement_depth(depth_img, rgb_img):
    pass


def min_max_scale_real(img, mask, lim=(55,200)): # object mask is a 56*56 bool matrix
   img = img.astype(np.float32)
   sorted_img = np.array(sorted(img[mask].reshape(-1,1)))
   min_5_percent = sorted_img[int(sorted_img.shape[0]*0.05)]
   max_5_percent = sorted_img[int(sorted_img.shape[0]*0.95)]
#    import pdb; pdb.set_trace()
   img_min = min_5_percent
   img_max = max_5_percent

   img[mask] = np.clip((img[mask]-img_min)/(img_max-img_min+1e-8)*(lim[1]-lim[0])+lim[0], 0, 255)
   img[mask] = 255-img[mask]
   img[~mask] = 0
   
   return img.astype(np.uint8)

def remove_noise(img): 
    xs, ys = np.where(img < 255)
    depth_img_r = np.zeros((np.shape(img)[0], np.shape(img)[1]))
    fitting_points = []
    for i in range(len(xs)): 
        fitting_points.append([xs[i], ys[i]])
    fitting_points = np.asarray(fitting_points)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=7).fit(fitting_points)
    labels = clustering.labels_
    for k in range(np.max(labels)+1): 
        if len(np.where(labels == k)[0]) > 10: 
            cluster_indexes = np.where(labels == k)[0]
            cluster_points = fitting_points[cluster_indexes]
            depth_img_r[cluster_points[:, 0], cluster_points[:, 1]] = img[cluster_points[:, 0], cluster_points[:, 1]]

    return depth_img_r

if __name__ == '__main__':
    import os
    root_path = '/home/zizhou/Diffusion_est/DATA/Real/large_v2/'
    ind = 1
    depth = cv2.imread(os.path.join(root_path, str(ind), '0_depth.png'), -1)

    ori = deepcopy(depth)


    kernel = np.ones((10, 10), np.uint8) 
    depth = cv2.dilate(depth, kernel) 
    depth = cv2.erode(depth, kernel+10) 
    depth = depth.astype(np.float32)
    # plt.imshow(depth)
    # plt.show()   
    for i in range(480):
        for j in range(640):
            l = (1-i/245+j/640)/2
            depth[i,j] -= int(l*24)
    plt.imshow(depth)
    plt.show()  
    for i in range(480): 
        for j in range(640): 
            if depth[i, j] < 200 or depth[i, j] > 630 : 
                depth[i, j] = 5000
    max_d = np.max(depth[depth<5000])
    min_d = np.min(depth[depth<5000])
    depth[depth<5000] = 205 * (depth[depth<5000] - min_d)/(max_d - min_d)
    depth[depth==5000] = 255

    # depth[depth<200] = 0
    # depth[depth>630] = 0

    plt.imshow(depth)
    plt.show()  

    depth = cv2.copyMakeBorder(depth.astype(np.uint8), top=80, bottom=80, left=0, right=0, borderType=cv2.BORDER_CONSTANT,value=255)
    depth = recenter_real(depth, None)
    depth = cv2.resize(depth, (56, 56), interpolation=cv2.INTER_NEAREST)

    plt.imshow(depth)
    plt.show()  

    plot_depth = deepcopy(depth)
    mask = find_mask(depth)

    processed_depth = std_scale(depth, mask, lim=(55,200))
    # processed_depth = min_max_scale_real(depth, mask, (90,200))
    processed_depth = np.clip(processed_depth.astype(np.float32) , 0, 255)
    processed_depth[~mask] = 0
    # processed_depth[processed_depth>245] = 0

    # reference = cv2.imread('/home/zizhou/Diffusion_est/DATA/20240822/episode_21997/depth1_v5m.png', -1)
    # plt.figure()
    # plt.imshow(reference+np.random.uniform(size=reference.shape)*4, cmap='gray')
    # plt.show()

    # import pdb; pdb.set_trace()
    plt.figure()
    # plt.imshow(ori)
    # plt.figure()
    # grey_img = np.zeros((56,56,3))
    # grey_img = processed_depth.astype(np.float32)
    # ori[:20,:10] = 600
    # import pdb; pdb.set_trace()
    plt.imshow(processed_depth.astype(np.float32), cmap='gray')
    plt.show()
    cv2.imwrite(os.path.join(root_path, 'test.png'), processed_depth)
