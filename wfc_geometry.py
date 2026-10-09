"""Build editable tile assets and a Geometry Nodes instancer for WFC results.

Layout created for each generated grid::

    WFC                (container collection, linked to the scene collection)
    |- WFC Grid        (point-mesh object with the "WFC Tiles" modifier)
    `- WFC Tiles       (collection of editable "Tile NNN" objects)

The tile collection is held by the Collection Info node of a per-grid node
group, so every grid keeps its own tile set. The instance
index of a point is its ``wfc_tile_id`` attribute, which selects the object at
that position in the tile collection; the builder rewrites the collection order
on every run so the position always equals the tile id.
"""

import bpy
from .wfc_appearance import sync as sync_appearance

from .wfc_solver import color_key


GROUP_NAME = "WFC Tile Instancer"
GROUP_PROP = "wfc_node_group"
COLLECTION_NODE = "WFC Tile Collection"
ATTRIBUTE_NAME = "wfc_tile_id"
TILE_PROP = "wfc_tile_id"
GRID_PROP = "wfc_grid"
STOCK_PROP = "wfc_stock"
COLOR_PROP = "wfc_color"
KEY_PROP = "wfc_tile_key"
MODIFIER_NAME = "WFC Tiles"
ROW_SPACING = 1.25


def _node_group(assets):
    group = bpy.data.node_groups.new(GROUP_NAME, "GeometryNodeTree")
    group[GROUP_PROP] = True
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    scale_socket = group.interface.new_socket(name="Tile Scale", in_out="INPUT", socket_type="NodeSocketFloat")
    scale_socket.default_value = 1.0
    scale_socket.min_value = 0.01
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = group.nodes
    links = group.links
    input_node = nodes.new("NodeGroupInput")
    input_node.location = (-600, 0)
    attribute = nodes.new("GeometryNodeInputNamedAttribute")
    attribute.data_type = "INT"
    attribute.inputs["Name"].default_value = ATTRIBUTE_NAME
    attribute.location = (-600, -260)
    collection = nodes.new("GeometryNodeCollectionInfo")
    collection.name = COLLECTION_NODE
    collection.label = "Tile Collection"
    collection.transform_space = "ORIGINAL"
    collection.inputs["Separate Children"].default_value = True
    collection.inputs["Reset Children"].default_value = True
    collection.inputs["Collection"].default_value = assets
    collection.location = (-350, -150)
    scale = nodes.new("ShaderNodeCombineXYZ")
    scale.location = (-350, -360)
    instance = nodes.new("GeometryNodeInstanceOnPoints")
    instance.inputs["Pick Instance"].default_value = True
    instance.location = (-80, 0)
    output = nodes.new("NodeGroupOutput")
    output.location = (180, 0)

    links.new(input_node.outputs["Geometry"], instance.inputs["Points"])
    links.new(collection.outputs["Instances"], instance.inputs["Instance"])
    links.new(attribute.outputs["Attribute"], instance.inputs["Instance Index"])
    for component in ("X", "Y", "Z"):
        links.new(input_node.outputs["Tile Scale"], scale.inputs[component])
    links.new(scale.outputs["Vector"], instance.inputs["Scale"])
    links.new(instance.outputs["Instances"], output.inputs["Geometry"])
    return group


def _collection_node(group):
    node = group.nodes.get(COLLECTION_NODE)
    if node is None or node.bl_idname != "GeometryNodeCollectionInfo":
        raise RuntimeError("Node group '%s' has no tile Collection Info node" % group.name)
    return node


def _grid_modifier(obj):
    """Return the WFC Geometry Nodes modifier of ``obj`` or None."""
    for modifier in obj.modifiers:
        group = getattr(modifier, "node_group", None)
        if modifier.type == "NODES" and group is not None and group.get(GROUP_PROP):
            return modifier
    return None


def find_assets(obj):
    """Return the tile collection bound to a generated grid, or None."""
    if obj is None or obj.type != "MESH":
        return None
    modifier = _grid_modifier(obj)
    if modifier is None:
        return None
    node = modifier.node_group.nodes.get(COLLECTION_NODE)
    if node is None or node.bl_idname != "GeometryNodeCollectionInfo":
        return None
    return node.inputs["Collection"].default_value


def is_grid(obj):
    """True for a mesh object carrying the WFC tile modifier."""
    return obj is not None and obj.type == "MESH" and _grid_modifier(obj) is not None


def _cube_mesh(name):
    mesh = bpy.data.meshes.new(name)
    vertices = [(x, y, z) for z in (0.0, 0.18) for y in (-0.5, 0.5) for x in (-0.5, 0.5)]
    faces = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4),
             (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return mesh


def _stock_material(index, color):
    material = bpy.data.materials.new("WFC Tile %03d" % index)
    material[STOCK_PROP] = True
    _paint(material, color)
    return material


def _paint(material, color):
    material.diffuse_color = color
    if not material.use_nodes:
        material.use_nodes = True
    shader = next((n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if shader is not None:
        shader.inputs["Base Color"].default_value = color
        shader.inputs["Roughness"].default_value = 0.75
    material[COLOR_PROP] = list(color)


def _sync_material(mesh, index, color):
    """Give a tile mesh without any material the stock color material.

    Existing materials, stock or user-made, are never touched.
    """
    if len(mesh.materials) == 0:
        mesh.materials.append(_stock_material(index, color))


def _normalize_colors(colors):
    result = []
    for color in colors:
        values = [float(v) for v in color]
        if len(values) == 3:
            values.append(1.0)
        if len(values) != 4:
            raise ValueError("Palette colors need 3 or 4 components")
        result.append(tuple(values))
    return result


def _tile_color(obj):
    """Palette color a tile object was made for, or None when unknown."""
    color = obj.get(COLOR_PROP)
    if color is None and obj.type == "MESH":
        # Tiles from earlier versions only carry the color on their stock material.
        stock = next((m for m in obj.data.materials if m is not None and m.get(STOCK_PROP)), None)
        color = stock.get(COLOR_PROP) if stock is not None else None
    if color is None or len(color) != 4:
        return None
    return tuple(float(v) for v in color)


def _tile_key(obj):
    """Identity a tile object was made for, or None when unknown.

    Tiles from earlier versions only carry a color, which is the key of a
    color-only tile.
    """
    key = obj.get(KEY_PROP)
    if isinstance(key, str):
        return key
    color = _tile_color(obj)
    return color_key(color) if color is not None else None


def _old_tiles(assets):
    """Tile objects of a collection in their previous id order."""
    tiles = [obj for obj in assets.objects if _tile_key(obj) is not None]
    tiles.sort(key=lambda obj: (obj.get(TILE_PROP) if isinstance(obj.get(TILE_PROP), int) else 1 << 30,
                                obj.name))
    return tiles


def _sync_tiles(assets, colors, keys):
    """Match old tiles to the new tile keys and rewrite ids and order.

    ``colors[i]`` is the center color of tile ``i`` and ``keys[i]`` its identity.
    Matched tiles keep their meshes, materials and transforms. Keys with no old
    tile get a new cube. Old tiles with no matching key are never deleted: they
    follow the active tiles with fresh ids so a later run can reuse them.
    """
    old = _old_tiles(assets)
    unmatched = list(old)
    tiles = []
    for index, (color, key) in enumerate(zip(colors, keys)):
        tile = next((obj for obj in unmatched if _tile_key(obj) == key), None)
        if tile is not None:
            unmatched.remove(tile)
        else:
            mesh = _cube_mesh("Tile %03d" % index)
            tile = bpy.data.objects.new("Tile %03d" % index, mesh)
            # Source tiles sit in a tidy row beside the grid; Reset Children ignores it.
            tile.location = ((index % 16) * ROW_SPACING, -2.0 - (index // 16) * ROW_SPACING, 0.0)
        tile[TILE_PROP] = index
        tile[COLOR_PROP] = list(color)
        tile[KEY_PROP] = key
        if tile.type == "MESH":
            _sync_material(tile.data, index, color)
        tiles.append(tile)

    for offset, obj in enumerate(unmatched):
        known = _tile_color(obj)
        obj[TILE_PROP] = len(colors) + offset
        obj[KEY_PROP] = _tile_key(obj)
        if known is not None:
            obj[COLOR_PROP] = list(known)

    keep = set(tiles) | set(unmatched)
    foreign = [obj for obj in assets.objects if obj not in keep]
    for obj in list(assets.objects):
        assets.objects.unlink(obj)
    for obj in tiles + unmatched + foreign:
        assets.objects.link(obj)

def _write_points(mesh, indices, width, height):
    mesh.clear_geometry()
    mesh.from_pydata([(x, y, 0.0) for y in range(height) for x in range(width)], [], [])
    mesh.update()
    attribute = mesh.attributes.get(ATTRIBUTE_NAME)
    if attribute is None or attribute.domain != "POINT" or attribute.data_type != "INT":
        if attribute is not None:
            mesh.attributes.remove(attribute)
        attribute = mesh.attributes.new(ATTRIBUTE_NAME, "INT", "POINT")
    attribute.data.foreach_set("value", indices)
    mesh.update()


def _in_scene(scene, collection):
    return collection == scene.collection or collection in scene.collection.children_recursive


def _validate(indices, colors, keys, width, height, existing):
    if width < 1 or height < 1:
        raise ValueError("Grid size must be at least 1x1")
    if len(indices) != width * height:
        raise ValueError("Expected %d tile indices, got %d" % (width * height, len(indices)))
    if len(colors) == 0:
        raise ValueError("No tiles to build")
    if len(keys) != len(colors) or len(set(keys)) != len(keys):
        raise ValueError("Tile keys must be unique and match the tile colors")
    if indices and (min(indices) < 0 or max(indices) >= len(colors)):
        raise ValueError("Tile index outside the tile list")
    if existing is not None:
        if not isinstance(existing, bpy.types.Object) or not is_grid(existing):
            raise ValueError("Existing object is not a generated WFC grid")
        if existing.mode != "OBJECT":
            raise RuntimeError("Leave Edit Mode before regenerating the WFC grid")
        _collection_node(_grid_modifier(existing).node_group)


def build(context, indices, colors, keys, width, height, existing=None, settings=None):
    """Create or regenerate a point grid and its tile assets.

    ``colors[i]`` and ``keys[i]`` describe tile ``i`` (center color and a unique
    identity, see ``wfc_solver.tile_types``). ``existing`` is a previously generated
    grid object; its tile meshes and materials are reused for matching keys.
    Returns ``(grid, assets)``. Tile IDs correspond to collection order.
    """
    indices = [int(i) for i in indices]
    colors = _normalize_colors(colors)
    keys = list(keys)
    _validate(indices, colors, keys, width, height, existing)

    scene = context.scene
    # Per-grid node group: its Collection Info node holds the tile collection.
    grid = existing
    assets = find_assets(grid) if grid is not None else None

    if grid is None:
        container = bpy.data.collections.new("WFC")
        scene.collection.children.link(container)
        mesh = bpy.data.meshes.new("WFC Grid")
        grid = bpy.data.objects.new("WFC Grid", mesh)
        container.objects.link(grid)
    parent = grid.users_collection[0] if grid.users_collection else scene.collection

    if assets is None:
        assets = bpy.data.collections.new("WFC Tiles")
    if not _in_scene(scene, assets):
        parent.children.link(assets)

    _sync_tiles(assets, colors, keys)

    if grid.data.users > 1 or grid.data.library is not None:
        # Never overwrite a mesh shared with another object.
        grid.data = grid.data.copy()
    _write_points(grid.data, indices, width, height)
    grid[GRID_PROP] = True

    modifier = _grid_modifier(grid)
    if modifier is None:
        modifier = grid.modifiers.new(MODIFIER_NAME, "NODES")
        modifier.node_group = _node_group(assets)
    else:
        # Regeneration keeps the node group; only re-point its collection.
        _collection_node(modifier.node_group).inputs["Collection"].default_value = assets
    grid.update_tag()
    if settings is not None:
        sync_appearance(context, parent, modifier.node_group, len(colors), settings)

    for obj in context.view_layer.objects:
        if obj.select_get():
            obj.select_set(False)
    if grid.name in context.view_layer.objects:
        grid.select_set(True)
        context.view_layer.objects.active = grid
    return grid, assets
