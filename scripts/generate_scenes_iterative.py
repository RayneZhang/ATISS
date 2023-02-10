"""Script used for iteratively generating and adjusting scenes using a previously trained model."""
import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
from operator import add

from training_utils import load_config
from utils import floor_plan_from_scene, export_scene, make_network_input_from_gen, print_predicted_labels

from scene_completion import poll_specific_class
from object_suggestions import sample_in_bbox

from scene_synthesis.datasets import filter_function, \
    get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network
from scene_synthesis.utils import get_textured_objects

from simple_3dviz import Scene
from simple_3dviz.window import show
from simple_3dviz.behaviours.keyboard import SnapshotOnKey, SortTriangles
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle
from simple_3dviz.behaviours.io import SaveFrames, SaveGif
from simple_3dviz.utils import render

from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh

def poll_generated_objects(dataset, current_boxes):
    """Show the objects in the current_scene and ask which ones to be
    removed."""
    classes = np.array(dataset.class_labels)
    labels = classes[current_boxes["class_labels"].argmax(-1)].tolist()[0]
    print(
        "The scene you selected contains {}".format(
            list(enumerate(labels))
        )
    )
    msg = "Enter the indices of objects to be removed, separated with commas\n"
    ois = [int(oi) for oi in input(msg).split(",") if oi != ""]
    idxs_kept = list(set(range(len(labels))) - set(ois))
    print("You are keeping the following indices {}".format(idxs_kept))

    return idxs_kept

def poll_moving_object(dataset, current_boxes):
    """Show the objects in the current_scene and ask which one to be
    moved."""
    classes = np.array(dataset.class_labels)
    labels = classes[current_boxes["class_labels"].argmax(-1)].tolist()[0]
    print(
        "The scene you selected contains {}".format(
            list(enumerate(labels))
        )
    )
    msg = "Enter the index of objects to be moved\n"
    index = int(input(msg))
    return index

def poll_specific_object():
    """Ask which object to be added."""
    msg = "Enter the object ID that you want to add to the scene\n"
    return input(msg)

def poll_vector(msg):
    """Ask the user to enter a vector based on the message (e.g., position, rotation, size)."""
    ois = [float(oi) for oi in input(msg).split(",") if oi != ""]
    return list(ois)

def get_object_by_path(id):
    """Get a textured object based on the provided object id."""
    #path = os.path.join("/3D-FRONT/3D-FUTURE-model/", id, "raw_object.obj")
    path = "/3D-FRONT/3D-FUTURE-model/" + id + "/raw_model.obj"
    # Load the furniture and scale it as it is given in the dataset
    raw_mesh = TexturedMesh.from_file(path)
    raw_mesh.scale([1,1,1])
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(0)
    R[0, 2] = -np.sin(0)
    R[2, 0] = np.sin(0)
    R[2, 2] = np.cos(0)
    R[1, 1] = 1.
    translation = [0,0,0]
    raw_mesh.affine_transform(R=R, t=translation)
    return raw_mesh

def main(argv):
    parser = argparse.ArgumentParser(
        description="Iteratively generate and adjust scenes using a previously trained model"
    )

    parser.add_argument(
        "--config_file",
        default="../config/bedrooms_config.yaml",
        help="Path to the file that contains the experiment configuration"
    )
    parser.add_argument(
        "--output_directory",
        default="../tester",
        help="Path to the output directory"
    )
    parser.add_argument(
        "--path_to_pickled_3d_futute_models",
        default="/tmp/threed_future_model_bedroom.pkl",
        help="Path to the 3D-FUTURE model meshes"
    )
    parser.add_argument(
        "--path_to_floor_plan_textures",
        default="../demo/floor_plan_texture_images",
        help="Path to floor texture images"
    )
    parser.add_argument(
        "--weight_file",
        default="../models/03QTRCQ2W/model_00050",
        help="Path to a pretrained model"
    )
    parser.add_argument(
        "--n_sequences",
        default=3,
        type=int,
        help="The number of sequences to be generated"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="1,1,1,1",
        help="Set the background of the scene"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,1,0",
        help="Up vector of the scene"
    )
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="-0.10923499,1.9325259,-7.19009",
        help="Camer position in the scene"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="512,512",
        help="Define the size of the scene and the window"
    )
    parser.add_argument(
        "--with_rotating_camera",
        action="store_true",
        help="Use a camera rotating around the object"
    )
    parser.add_argument(
        "--save_frames",
        help="Path to save the visualization frames to"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=360,
        help="Number of frames to be rendered"
    )
    parser.add_argument(
        "--without_screen",
        action="store_true",
        help="Perform no screen rendering"
    )
    parser.add_argument(
        "--scene_id",
        default=None,
        help="The scene id to be used for conditioning"
    )

    args = parser.parse_args(argv)

    # Disable trimesh's logger
    logging.getLogger("trimesh").setLevel(logging.ERROR)

    # Option 1: running on GPU
    # if torch.cuda.is_available():
    #     device = torch.device("cuda:0")
    # else:
    #     device = torch.device("cpu")
    
    # Option 2: running on CPU because GPU might run out of memory.
    device = torch.device("cpu")
    print("Running code on", device)

    # Check if output directory exists and if it doesn't create it
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    config = load_config(args.config_file)

    raw_dataset, train_dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["training"].get("splits", ["train", "val"])
        ),
        split=config["training"].get("splits", ["train", "val"])
    )

    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_futute_models
    )
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(
            config["data"],
            split=config["validation"].get("splits", ["test"])
        ),
        split=config["validation"].get("splits", ["test"])
    )
    print("Loaded {} scenes with {} object types:".format(
        len(dataset), dataset.n_object_types)
    )

    network, _, _ = build_network(
        dataset.feature_size, dataset.n_classes,
        config, args.weight_file, device=device
    )
    network.eval()

    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(size=args.window_size)
    scene.up_vector = args.up_vector
    scene.camera_target = args.camera_target
    scene.camera_position = args.camera_position
    scene.light = args.camera_position

    given_scene_id = None

    classes = np.array(dataset.class_labels)
    scene_idx = given_scene_id or np.random.choice(len(dataset))
    current_scene = raw_dataset[scene_idx]
    bbox_params_all = []
    for i in range(args.n_sequences):
        print("{} / {}: Using the floor plan of scene {}".format(
            i, args.n_sequences, current_scene.scene_id)
        )
        # Get a floor plan from a selected "test" scene
        floor_plan, tr_floor, room_mask = floor_plan_from_scene(
            current_scene, args.path_to_floor_plan_textures
        )

        bbox_params = network.generate_boxes(
            room_mask=room_mask.to(device),
            device=device
        )
        bbox_params_all.append(bbox_params)
        
        boxes = dataset.post_process(bbox_params)
        bbox_params_t = torch.cat([
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ], dim=-1).cpu().numpy()
        renderables, trimesh_meshes = get_textured_objects(
            bbox_params_t, objects_dataset, classes
        )
        renderables += floor_plan
        trimesh_meshes += tr_floor

        if args.without_screen:
            # Do the rendering
            path_to_image = "{}/{}_{}_{:03d}".format(
                args.output_directory,
                current_scene.scene_id,
                scene_idx,
                i
            )
            behaviours = [
                LightToCamera(),
                SaveFrames(path_to_image+".png", 1)
            ]
            if args.with_rotating_camera:
                behaviours += [
                    CameraTrajectory(
                        Circle(
                            [0, args.camera_position[1], 0],
                            args.camera_position,
                            args.up_vector
                        ),
                        speed=1/360
                    ),
                    SaveGif(path_to_image+".gif", 1)
                ]

            render(
                renderables,
                behaviours=behaviours,
                size=args.window_size,
                camera_position=args.camera_position,
                camera_target=args.camera_target,
                up_vector=args.up_vector,
                background=args.background,
                n_frames=args.n_frames,
                scene=scene
            )
        else:
            show(
                renderables,
                behaviours=[LightToCamera(), SnapshotOnKey(), SortTriangles()],
                size=args.window_size,
                camera_position=args.camera_position,
                camera_target=args.camera_target,
                up_vector=args.up_vector,
                background=args.background,
                title="Generated Scene {}".format(i)
            )

    bbox_params = bbox_params_all[int(input("Which room do you want? Type its index (0-2)\n"))]
    print(bbox_params)
    while input("Do you want to ask for an object suggestion? Type 'yes' or 'no'\n") == "yes":
        object_indices = poll_generated_objects(dataset, bbox_params)
        bbox_bounds = [
            float(ti)
            for ti in input("Enter bbox dims to place an object\n").split(",")
        ]
        boxes = make_network_input_from_gen(bbox_params, object_indices)
        query_class_label = poll_specific_class(dataset)

        # Given the current context predict the probability of all class labels
        with torch.no_grad():
            class_probs = network.distribution_classes(
                boxes=boxes, room_mask=room_mask
            )
            translation_probs = [
                network.distribution_translations(boxes, room_mask, c)
                for c in range(len(dataset.object_types))
            ]
        try:
            new_class, (tx, ty, tz) = sample_in_bbox(class_probs,
                                                     translation_probs,
                                                     bbox_bounds, 100000, query_class_label)
        except RuntimeError:
            continue
        print("Adding {} at location: ({:.4f}, {:.4f}, {:.4f})".format(
            dataset.object_types[new_class], tx, ty, tz
        ))

        bbox_params = network.add_object_with_class_and_translation(
            boxes,
            room_mask,
            int(new_class),
            torch.tensor([tx, ty, tz])[None, None].float()
        )

        boxes = dataset.post_process(bbox_params)
        bbox_params_t = torch.cat([
            boxes["class_labels"],
            boxes["translations"],
            boxes["sizes"],
            boxes["angles"]
        ], dim=-1).cpu().numpy()
        renderables, trimesh_meshes = get_textured_objects(
            bbox_params_t, objects_dataset, classes
        )
        renderables += floor_plan
        trimesh_meshes += tr_floor

        show(
            renderables,
            behaviours=[LightToCamera(), SnapshotOnKey(), SortTriangles()],
            size=args.window_size,
            camera_position=args.camera_position,
            camera_target=args.camera_target,
            up_vector=args.up_vector,
            background=args.background,
            title="Suggested Object Scene"
        )

    while input("Are you satisfied with this design? Type 'yes' or 'no'\n") != "yes":
        query_mesh = None
        if input("Do you want to add an object? Type 'yes' or 'no'\n") == "yes":
            query_object_id = poll_specific_object()
            if query_object_id is not None:
                print("You are trying to add the object {}".format(query_object_id))
                query_mesh = get_object_by_path(query_object_id)
                print(query_mesh)
            pos = [0,0,0]
            while input("Do you want to move this object? Type 'yes' or 'no'\n") == "yes":
                offset = poll_vector("Type the offset in the format of (x,y,z):")
                pos = list(map(add, pos, offset))
                print(pos)
                query_mesh.affine_transform(t=np.array(pos))
                boxes = dataset.post_process(bbox_params)
                bbox_params_t = torch.cat(
                    [
                        boxes["class_labels"],
                        boxes["translations"],
                        boxes["sizes"],
                        boxes["angles"]
                    ],
                    dim=-1
                ).cpu().numpy()
                renderables, trimesh_meshes = get_textured_objects(
                    bbox_params_t, objects_dataset, classes
                )
                if query_mesh is not None:
                    renderables.append(query_mesh)
                renderables += floor_plan
                trimesh_meshes += tr_floor

                show(
                renderables,
                behaviours=[LightToCamera(), SnapshotOnKey(), SortTriangles()],
                size=args.window_size,
                camera_position=args.camera_position,
                camera_target=args.camera_target,
                up_vector=args.up_vector,
                background=args.background,
                title="Re-generated Scene"
                )

        while input("Do you want to move generated objects? Type 'yes' or 'no'\n") == "yes":
            object_index = poll_moving_object(dataset, bbox_params)
            offset = poll_vector("Type the offset in the format of (x,y,z):")
            #print(bbox_params["translations"])
            object_position = bbox_params["translations"][0][object_index]
            bbox_params["translations"][0][object_index] =  torch.add(object_position, torch.Tensor(offset))
            #print(bbox_params["translations"])
            boxes = dataset.post_process(bbox_params)
            bbox_params_t = torch.cat([
                boxes["class_labels"],
                boxes["translations"],
                boxes["sizes"],
                boxes["angles"]
            ], dim=-1).cpu().numpy()
            renderables, trimesh_meshes = get_textured_objects(
                bbox_params_t, objects_dataset, classes
            )
            renderables += floor_plan
            trimesh_meshes += tr_floor

            show(
                renderables,
                behaviours=[LightToCamera(), SnapshotOnKey(), SortTriangles()],
                size=args.window_size,
                camera_position=args.camera_position,
                camera_target=args.camera_target,
                up_vector=args.up_vector,
                background=args.background,
                title="Moved Scene"
            )

        object_indices = poll_generated_objects(dataset, bbox_params)
        input_boxes = make_network_input_from_gen(bbox_params, object_indices)

        completion_config = load_config("../config/bedrooms_eval_config.yaml")
        # New network for scene completion
        completion_network, _, _ = build_network(
            dataset.feature_size, dataset.n_classes,
            completion_config, args.weight_file, device=device
        )
        completion_network.eval()

        query_class_label = poll_specific_class(dataset)
        if query_class_label is not None:
            print("Adding a single object")
            bbox_params = completion_network.add_object(
                room_mask=room_mask,
                class_label=query_class_label,
                boxes=input_boxes
            )
        else:
            print("Doing scene completion")
            bbox_params = completion_network.complete_scene(
                boxes=input_boxes, room_mask=room_mask
            )

        boxes = dataset.post_process(bbox_params)
        bbox_params_t = torch.cat(
            [
                boxes["class_labels"],
                boxes["translations"],
                boxes["sizes"],
                boxes["angles"]
            ],
            dim=-1
        ).cpu().numpy()
        print_predicted_labels(dataset, boxes)

        renderables, trimesh_meshes = get_textured_objects(
            bbox_params_t, objects_dataset, classes
        )
        if query_mesh is not None:
            renderables.append(query_mesh)
        renderables += floor_plan
        trimesh_meshes += tr_floor

        show(
        renderables,
        behaviours=[LightToCamera(), SnapshotOnKey(), SortTriangles()],
        size=args.window_size,
        camera_position=args.camera_position,
        camera_target=args.camera_target,
        up_vector=args.up_vector,
        background=args.background,
        title="Re-generated Scene"
        )

        if trimesh_meshes is not None:
            # Create a trimesh scene and export it
            path_to_objs = os.path.join(
                args.output_directory,
                "{:03d}_scene".format(i)
            )
            if not os.path.exists(path_to_objs):
                os.mkdir(path_to_objs)
            export_scene(path_to_objs, trimesh_meshes)

if __name__ == "__main__":
    main(sys.argv[1:])
