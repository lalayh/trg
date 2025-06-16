import os
import trimesh
from urdfpy import URDF


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    The returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces, visual=g.visual)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def get_mesh_pose_list_from_world(world, object_set, exclude_plane=True):
    mesh_pose_list = []
    # collect object mesh paths and poses
    for uid in world.bodies.keys():
        _, name = world.p.getBodyInfo(uid)
        name = name.decode('utf8')
        if name == 'plane' and exclude_plane:
            continue
        body = world.bodies[uid]
        pose = body.get_pose().as_matrix()
        scale = body.scale
        visuals = world.p.getVisualShapeData(uid)
        assert len(visuals) == 1
        _, _, _, _, mesh_path, _, _, _ = visuals[0]
        mesh_path = mesh_path.decode('utf8')
        if mesh_path == '':
            mesh_path = os.path.join('./data/urdfs', object_set, name + '.urdf')
        mesh_pose_list.append((mesh_path, scale, pose))
    return mesh_pose_list


def get_scene_from_mesh_pose_list(mesh_pose_list, scene_as_mesh=True, return_list=False):
    # create scene from meshes
    scene = trimesh.Scene()
    mesh_list = []
    for mesh_path, scale, pose in mesh_pose_list:
        if os.path.splitext(mesh_path)[1] == '.urdf':
            obj = URDF.load(mesh_path)
            assert len(obj.links) == 1
            assert len(obj.links[0].visuals) == 1
            assert len(obj.links[0].visuals[0].geometry.meshes) == 1
            mesh = obj.links[0].visuals[0].geometry.meshes[0].copy()
        else:
            mesh = trimesh.load(mesh_path)

        mesh.apply_scale(scale)
        mesh.apply_transform(pose)
        scene.add_geometry(mesh)
        mesh_list.append(mesh)
    if scene_as_mesh:
        scene = as_mesh(scene)
    if return_list:
        return scene, mesh_list
    else:
        return scene
