import numpy as np
import random
import sys
import bpy


def blender_image_to_modulelist(b3d_image):
    img_source_w = b3d_image.size[0]
    img_source_h = b3d_image.size[1]

    # Pixel target image array
    pixel_target = np.full((img_source_h, img_source_w), "_9__9__9_")

    # Get all image pixels
    img_source_array = b3d_image.pixels[:]

    for height_i in range(0, img_source_h):

        for width_i in range(0, img_source_w):
            print(width_i)
            # Get pixel position in flat array
            colar = (1 + (width_i + (height_i * img_source_w))) * 4

            # Get color values at current "Pixel"
            r = str(round(img_source_array[colar - 4], 1))
            g = str(round(img_source_array[colar - 3], 1))
            b = str(round(img_source_array[colar - 2], 1))
            rgb = r+g+b

            # Append a list of the rgb values for each pixel to the line list
            pixel_target[height_i][width_i] = rgb

    # Empty target module array
    module_target = np.full((img_source_h-2, img_source_w-2),
                            "45_45_45_45_45_45_45_45_45_45_45_45_45_45_45_")
    for height_i in range(1, img_source_h-1):

        for width_i in range(1, img_source_w-1):

            # Return a list of neighbours
            current_module = pixelstr_neighbours_list(
                height_i, width_i, pixel_target)
            module_target[height_i-1][width_i-1] = current_module

    return module_target


def pixelstr_neighbours_list(x_coord, y_coord, pixel_list):

    # Append neighbours
    zero = pixel_list[x_coord][y_coord]
    xminus = pixel_list[x_coord-1][y_coord]
    yplus = pixel_list[x_coord][y_coord+1]
    xplus = pixel_list[x_coord + 1][y_coord]
    yminus = pixel_list[x_coord][y_coord-1]
    neighbours_list = zero + xminus + yplus + xplus + yminus
    return neighbours_list


def place_blocks(module_list):

    # Find Collection
    seed = random.randrange(sys.maxsize)
    result_name = 'Result ' + str(seed)
    unique_tiles = np.unique(module_list)

    # Find Tile Collection or create one
    if (bpy.data.collections.find('Tiles') >= 0):
        tiles_collection = bpy.data.collections['Tiles']
    else:
        tiles_collection = bpy.data.collections.new('Tiles')
        bpy.context.scene.collection.children.link(tiles_collection)

    # Create all unique tiles
    for t in range(0, len(unique_tiles)):
        if bpy.data.collections['Tiles'].objects.find(unique_tiles[t]) >= 0:
            pass
        else:
            bpy.ops.mesh.primitive_plane_add(location=(t, -5, 0), size=1)

            basemesh = bpy.context.object

            # if bpy.data.collections['Tiles'].objects.find(basemesh.name) >= 0:
            tiles_collection.objects.link(basemesh)

            bpy.data.collections['Collection'].objects.unlink(basemesh)

            basemesh.name = unique_tiles[t]

    # Create result collection
    result_collection = bpy.data.collections.new(result_name)

    bpy.context.scene.collection.children.link(result_collection)

    # Copy tile and change the respective position
    for x in range(0, len(module_list)):

        for y in range(0, len(module_list[x])):

            srcobj = tiles_collection.objects[module_list[x][y]]

            if srcobj.type == 'MESH':

                print('mesh')
                copymesh = srcobj.copy()
                copymesh.name = module_list[x][y] + str((x + 1) * (y + 1))
                copymesh.data = srcobj.data

                # copymesh = bpy.context.object
                copymesh.location = (-x, y, 0)
                result_collection.objects.link(copymesh)
            else:
                pass

    # set mesh name

    return


class WFC_OT_Placer(bpy.types.Operator):
    bl_idname = "object.wfc_ot_placer"
    bl_label = "Place"

    def execute(self, context):
        img_name = bpy.context.scene.wfc_vars.wfc_images_placer
        img = bpy.data.images[img_name]
        module_list = blender_image_to_modulelist(img)
        place_blocks(module_list)
        return {'FINISHED'}
