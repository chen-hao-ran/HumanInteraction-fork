# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import trimesh
import pyrender
import numpy as np
import colorsys
import cv2


class Renderer(object):

    def __init__(self, focal_length=600, img_w=512, img_h=512, faces=None,
                 same_mesh_color=False):
        # os.environ['PYOPENGL_PLATFORM'] = 'egl'
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w,
                                                   viewport_height=img_h,
                                                   point_size=1.0)
        self.camera_center = [img_w // 2, img_h // 2]
        self.focal_length = focal_length
        self.faces = faces
        self.same_mesh_color = same_mesh_color

    def render_front_view(self, verts, contact_label, bg_img_rgb=None, bg_color=(0, 0, 0, 0)):
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=np.ones(3) * 0)
        # Create camera. Camera will always be at [0,0,0]
        camera = pyrender.camera.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                                  cx=self.camera_center[0], cy=self.camera_center[1], zfar=1e12)
        scene.add(camera, pose=np.eye(4))

        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        # for DirectionalLight, only rotation matters
        light_pose = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0])
        scene.add(light, pose=light_pose)
        light_pose = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
        scene.add(light, pose=light_pose)

        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        # multiple person
        num_people = len(verts)
        # for every person in the scene
        for n in range(num_people):
            n_vertices = len(verts[n])
            vertex_colors = np.ones((n_vertices, 4)) * [0.0, 1.0, 0.0, 1.0]
            import json
            import pickle
            # c_label = contact_label[n]
            # rid_to_vid_smpl = pickle.load(open("output/rid_to_vid_smpl.pkl", "rb"))
            # c_ind = np.where(c_label == 1)[0]
            # for c_rid in c_ind:
            #     vids = rid_to_vid_smpl[c_rid]
            #     vertex_colors[vids] = np.ones(4) * [0.0, 0.0, 1.0, 1.0]

            # c_label = contact_label[n]
            # c_ind = np.where(c_label > 0)[0]
            # vertex_colors[c_ind] = np.ones(4) * [0.0, 0.0, 1.0, 1.0]

            c_label = contact_label[n]
            rid_to_vid_smpl = np.load("output/vid_to_rid_smpl.npy")
            c_ind = np.where(c_label == 1)[0]
            for c_rid in c_ind:
                v_ind = np.where(rid_to_vid_smpl == c_rid)[0]
                vertex_colors[v_ind] = np.ones(4) * [0.0, 0.0, 1.0, 1.0]

            mesh = trimesh.Trimesh(verts[n], self.faces, vertex_colors=vertex_colors)
            mesh.apply_transform(rot)
            mesh = pyrender.Mesh.from_trimesh(mesh)
            scene.add(mesh, 'mesh')

        # Alpha channel was not working previously, need to check again
        # Until this is fixed use hack with depth image to get the opacity
        color_rgba, depth_map = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color_rgb = color_rgba[:, :, :3]
        if bg_img_rgb is None:
            return color_rgb
        else:
            mask = depth_map > 0
            bg_img_rgb[mask] = color_rgb[mask]
            return bg_img_rgb

    def render_side_view(self, verts, contact_label):
        centroid = verts.mean(axis=(0, 1))  # n*6890*3 -> 3
        # make the centroid at the image center (the X and Y coordinates are zeros)
        centroid[:2] = 0
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0][np.newaxis, ...]  # 1*3*3
        pred_vert_arr_side = np.matmul((verts - centroid), aroundy) + centroid
        side_view = self.render_front_view(pred_vert_arr_side, contact_label)
        return side_view

    def render_back_view(self, verts):
        centroid = verts.mean(axis=(0, 1))  # n*6890*3 -> 3
        # make the centroid at the image center (the X and Y coordinates are zeros)
        centroid[:2] = 0
        aroundy = cv2.Rodrigues(np.array([0, np.radians(180.), 0]))[0][np.newaxis, ...]  # 1*3*3
        pred_vert_arr_side = np.matmul((verts - centroid), aroundy) + centroid
        back_view = self.render_front_view(pred_vert_arr_side)
        return back_view

    def delete(self):
        """
        Need to delete before creating the renderer next time
        """
        self.renderer.delete()
