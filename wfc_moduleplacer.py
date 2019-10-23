import numpy as np
import random
import sys
import bpy

# PART BELOW NOT WORKING TAKES CHANNEL VALUES INSTEAD OF PIXEL VALUES


def get_tiles(tiles):
    temp_tiles = np.unique(tiles)
    # print(temp_tiles)
    # l_len = len(tiles[0])
    # for x in range(0, len(tiles)):
    #     for y in range(0, l_len):
    #         if (tiles == temp_tiles).all():
    #             temp_tiles.append(tiles[x][y])
    #         # if tiles[x][y] not in temp_tiles:
    #         #     temp_tiles.append(tiles[x][y])
    #         else:
    #             pass
    return temp_tiles


def block_placer(output_array):
    # Find Collection
    seed = random.randrange(sys.maxsize)
    result_name = 'Result ' + str(seed)

    if (bpy.data.collections.find('Tiles') == True):
        tiles_collection = bpy.data.collections['Tiles']
    else:
        tiles_collection = bpy.data.collections.new('Tiles')

    bpy.context.scene.collection.children.link(tiles_collection)
    l_len = len(output_array[0])
    variants = get_tiles(output_array)
    result_collection = bpy.data.collections.new(result_name)
    bpy.context.scene.collection.children.link(result_collection)
    for t in range(0, len(variants)):
        bpy.ops.mesh.primitive_plane_add(location=(t, -5, 0), size=1)
        basemesh = bpy.context.object
        basemesh.name = str(variants[t])
        # bpy.data.collections['Tiles'].objects.link(basemesh)
        tiles_collection.objects.link(basemesh)
        # bpy.context.scene.collection.children.unlink(basemesh)
        # print(t)

    for x in range(0, len(output_array)):
            # in the cell
        for y in range(0, l_len):
            # print(output_array[0][x][y])
            srcobj = tiles_collection.objects[output_array[x][y]]
            if srcobj.type == 'MESH':
                print('mesh')
                copymesh = srcobj.copy()
                copymesh.name = output_array[x][y] + str((x + 1) * (y + 1))
                copymesh.data = srcobj.data
                # copymesh = bpy.context.object
                copymesh.location = (x, y, 0)
                result_collection.objects.link(copymesh)
            else:
                pass

        # set mesh name
    return


def blender_transitions(wfc_result_array):
    output_tiles = []
    for x in range(0, len(wfc_result_array)):
        l_len = len(wfc_result_array[0])
        for y in range(0, l_len):
            blender_tile = ['L', 'L', 'L', 'L']
            if x - 1 >= 0:
                blender_tile[0] = wfc_result_array[x-1][y]
            if y + 1 < l_len:
                blender_tile[1] = wfc_result_array[x][y+1]
            if x + 1 < len(wfc_result_array):
                blender_tile[2] = wfc_result_array[x+1][y]
            if y - 1 >= 0:
                blender_tile[3] = wfc_result_array[x][y-1]
            output_tiles.append(blender_tile)

    return output_tiles


class WFC_OT_Placer(bpy.types.Operator):
    bl_idname = "object.wfc_ot_placer"
    bl_label = "Place"

    def execute(self, context):

        return {'FINISHED'}
