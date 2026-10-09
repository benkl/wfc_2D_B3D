"""Visualise which tiles touch which on each side, driven by Geometry Nodes.

For a generated grid this builds a second object, ``WFC Relations``. Every tile
that occurs in the grid gets one block: the tile sits in the middle and each tile
seen next to it on the right, up, left or down side is instanced on that side and
joined to the centre by a line. Instances come from the grid's own tile collection,
so edits to tile meshes show up here as well. Right is +X and up is +Y, as in the
grid itself.

The mesh stores one vertex per tile (``wfc_rel_side`` = -1) and one per relation,
all at the block centre. Four integer point attributes drive the node group:

* ``wfc_tile_id``   tile instanced at the point (centre tile or neighbour)
* ``wfc_rel_side``  -1 for the centre, else 0 right, 1 up, 2 left, 3 down
* ``wfc_rel_slot``  position of the neighbour among those on the same side
* ``wfc_rel_count`` number of neighbours on that side

The node group turns side, slot and count into an offset, shrinks neighbours, and
turns the edges back to the centre into tubes. Relations are the ones observed in
the solved grid, not every pairing the source image could allow.
"""

import math

import bpy

from .wfc_geometry import ATTRIBUTE_NAME, COLLECTION_NODE, find_assets, is_grid
from .wfc_solver import tile_relations

GROUP_NAME = "WFC Relations"
GROUP_PROP = "wfc_relations_group"
OBJECT_PROP = "wfc_relations"      # grid -> relations object
SOURCE_PROP = "wfc_relations_of"   # relations object -> grid
MODIFIER_NAME = "WFC Relations"
SIDE_ATTRIBUTE = "wfc_rel_side"
SLOT_ATTRIBUTE = "wfc_rel_slot"
COUNT_ATTRIBUTE = "wfc_rel_count"

DISTANCE = 1.6   # centre tile to the neighbour row
SPACING = 0.75   # gap between neighbours on the same side
TILE_SCALE = 0.8
NEIGHBOUR_SCALE = 0.6
LINE_RADIUS = 0.02
_BLOCK_MARGIN = 0.8


def find_relations(grid):
    """The relations object of a grid, or None."""
    obj = grid.get(OBJECT_PROP)
    if isinstance(obj, bpy.types.Object) and obj.get(SOURCE_PROP) == grid:
        return obj
    return None


def grid_tiles(grid):
    """Tile ids and the size of a generated grid object."""
    mesh = grid.data
    attribute = mesh.attributes.get(ATTRIBUTE_NAME)
    count = len(mesh.vertices)
    if attribute is None or attribute.domain != "POINT" or attribute.data_type != "INT" or count == 0:
        raise ValueError("Grid has no wfc_tile_id point attribute")
    ids = [0] * count
    attribute.data.foreach_get("value", ids)
    coordinates = [0.0] * (count * 3)
    mesh.vertices.foreach_get("co", coordinates)
    width = int(round(max(coordinates[0::3]))) + 1
    if count % width:
        raise ValueError("Grid points do not form a rectangle")
    return ids, width, count // width


def layout(tile_indices, width, height):
    """Mesh data for the relations view of a solved grid.

    Returns ``(positions, edges, tile, side, slot, count)`` with one entry per
    vertex. Centre vertices come first, ordered by tile id.
    """
    relations = tile_relations(tile_indices, width, height)
    tiles = sorted(set(tile_indices))
    centre_of = {t: i for i, t in enumerate(tiles)}
    per_side = {}
    for tile, side, other in relations:
        per_side.setdefault((tile, side), []).append(other)
    widest = max((len(v) for v in per_side.values()), default=1)

    reach = max(DISTANCE, (widest - 1) * SPACING / 2.0) + _BLOCK_MARGIN
    pitch = 2.0 * reach + 0.4
    columns = max(1, int(math.ceil(math.sqrt(len(tiles)))))

    positions = [((i % columns) * pitch, -(i // columns) * pitch, 0.0) for i in range(len(tiles))]
    tile_attr = list(tiles)
    side_attr = [-1] * len(tiles)
    slot_attr = [0] * len(tiles)
    count_attr = [0] * len(tiles)
    edges = []
    for (tile, side), others in sorted(per_side.items()):
        centre = centre_of[tile]
        for slot, other in enumerate(others):
            edges.append((centre, len(positions)))
            positions.append(positions[centre])
            tile_attr.append(other)
            side_attr.append(side)
            slot_attr.append(slot)
            count_attr.append(len(others))
    return positions, edges, tile_attr, side_attr, slot_attr, count_attr


def _write_mesh(mesh, data):
    positions, edges, tile, side, slot, count = data
    mesh.clear_geometry()
    mesh.from_pydata(positions, edges, [])
    mesh.update()
    for name, values in ((ATTRIBUTE_NAME, tile), (SIDE_ATTRIBUTE, side),
                         (SLOT_ATTRIBUTE, slot), (COUNT_ATTRIBUTE, count)):
        attribute = mesh.attributes.get(name)
        if attribute is None or attribute.domain != "POINT" or attribute.data_type != "INT":
            if attribute is not None:
                mesh.attributes.remove(attribute)
            attribute = mesh.attributes.new(name, "INT", "POINT")
        attribute.data.foreach_set("value", values)
    mesh.update()


def _socket(group, name, default, minimum):
    socket = group.interface.new_socket(name=name, in_out="INPUT", socket_type="NodeSocketFloat")
    socket.default_value = default
    socket.min_value = minimum


def _math(group, op, a, b=None, c=None, x=0.0, y=0.0):
    """Add a Math node; sockets are linked, plain numbers become defaults."""
    node = group.nodes.new("ShaderNodeMath")
    node.operation = op
    node.location = (x, y)
    for target, value in zip(node.inputs, (a, b, c)):
        if isinstance(value, bpy.types.NodeSocket):
            group.links.new(value, target)
        elif value is not None:
            target.default_value = value
    return node.outputs[0]


def _node_group(assets):
    group = bpy.data.node_groups.new(GROUP_NAME, "GeometryNodeTree")
    group[GROUP_PROP] = True
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    _socket(group, "Tile Scale", TILE_SCALE, 0.01)
    _socket(group, "Neighbour Scale", NEIGHBOUR_SCALE, 0.01)
    _socket(group, "Distance", DISTANCE, 0.0)
    _socket(group, "Spacing", SPACING, 0.0)
    _socket(group, "Line Radius", LINE_RADIUS, 0.0)
    group.interface.new_socket(name="Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = group.nodes
    links = group.links
    inp = nodes.new("NodeGroupInput")
    inp.location = (-1500, 0)
    out = nodes.new("NodeGroupOutput")
    out.location = (900, 0)

    def attribute(name, y):
        node = nodes.new("GeometryNodeInputNamedAttribute")
        node.data_type = "INT"
        node.inputs["Name"].default_value = name
        node.location = (-1500, y)
        return node.outputs["Attribute"]

    tile = attribute(ATTRIBUTE_NAME, -200)
    side = attribute(SIDE_ATTRIBUTE, -330)
    slot = attribute(SLOT_ATTRIBUTE, -460)
    count = attribute(COUNT_ATTRIBUTE, -590)

    # The centre tile has side -1: it gets no offset and keeps full size.
    has_side = _math(group, "GREATER_THAN", side, -0.5, x=-1250, y=-300)
    angle = _math(group, "MULTIPLY", side, math.pi / 2.0, x=-1250, y=-420)
    cos_a = _math(group, "COSINE", angle, x=-1050, y=-380)
    sin_a = _math(group, "SINE", angle, x=-1050, y=-480)
    along = _math(group, "MULTIPLY", has_side, inp.outputs["Distance"], x=-1050, y=-250)
    middle = _math(group, "MULTIPLY_ADD", count, 0.5, -0.5, x=-1250, y=-560)
    centred = _math(group, "SUBTRACT", slot, middle, x=-1050, y=-600)
    across = _math(group, "MULTIPLY", centred, inp.outputs["Spacing"], x=-850, y=-600)
    across = _math(group, "MULTIPLY", across, has_side, x=-650, y=-600)
    x_off = _math(group, "SUBTRACT",
                  _math(group, "MULTIPLY", cos_a, along, x=-850, y=-300),
                  _math(group, "MULTIPLY", sin_a, across, x=-850, y=-420), x=-650, y=-360)
    y_off = _math(group, "ADD",
                  _math(group, "MULTIPLY", sin_a, along, x=-850, y=-500),
                  _math(group, "MULTIPLY", cos_a, across, x=-850, y=-700), x=-650, y=-520)
    offset = nodes.new("ShaderNodeCombineXYZ")
    offset.location = (-450, -400)
    links.new(x_off, offset.inputs["X"])
    links.new(y_off, offset.inputs["Y"])

    shrink = _math(group, "SUBTRACT", 1.0, inp.outputs["Neighbour Scale"], x=-1050, y=-150)
    factor = _math(group, "SUBTRACT", 1.0,
                   _math(group, "MULTIPLY", has_side, shrink, x=-850, y=-150), x=-650, y=-150)
    uniform = _math(group, "MULTIPLY", factor, inp.outputs["Tile Scale"], x=-450, y=-150)
    scale = nodes.new("ShaderNodeCombineXYZ")
    scale.location = (-250, -150)
    for component in ("X", "Y", "Z"):
        links.new(uniform, scale.inputs[component])

    place = nodes.new("GeometryNodeSetPosition")
    place.location = (-250, 100)
    links.new(inp.outputs["Geometry"], place.inputs["Geometry"])
    links.new(offset.outputs["Vector"], place.inputs["Offset"])

    collection = nodes.new("GeometryNodeCollectionInfo")
    collection.name = COLLECTION_NODE
    collection.label = "Tile Collection"
    collection.transform_space = "ORIGINAL"
    collection.inputs["Separate Children"].default_value = True
    collection.inputs["Reset Children"].default_value = True
    collection.inputs["Collection"].default_value = assets
    collection.location = (-250, -330)

    instance = nodes.new("GeometryNodeInstanceOnPoints")
    instance.inputs["Pick Instance"].default_value = True
    instance.location = (50, 100)
    links.new(place.outputs["Geometry"], instance.inputs["Points"])
    links.new(collection.outputs["Instances"], instance.inputs["Instance"])
    links.new(tile, instance.inputs["Instance Index"])
    links.new(scale.outputs["Vector"], instance.inputs["Scale"])

    to_curve = nodes.new("GeometryNodeMeshToCurve")
    to_curve.location = (50, -150)
    links.new(place.outputs["Geometry"], to_curve.inputs["Mesh"])
    circle = nodes.new("GeometryNodeCurvePrimitiveCircle")
    circle.inputs["Resolution"].default_value = 6
    circle.location = (50, -330)
    links.new(inp.outputs["Line Radius"], circle.inputs["Radius"])
    tube = nodes.new("GeometryNodeCurveToMesh")
    tube.location = (300, -150)
    links.new(to_curve.outputs["Curve"], tube.inputs["Curve"])
    links.new(circle.outputs["Curve"], tube.inputs["Profile Curve"])

    join = nodes.new("GeometryNodeJoinGeometry")
    join.location = (600, 0)
    links.new(instance.outputs["Instances"], join.inputs["Geometry"])
    links.new(tube.outputs["Mesh"], join.inputs["Geometry"])
    links.new(join.outputs["Geometry"], out.inputs["Geometry"])
    return group


def _modifier(obj):
    for modifier in obj.modifiers:
        group = getattr(modifier, "node_group", None)
        if modifier.type == "NODES" and group is not None and group.get(GROUP_PROP):
            return modifier
    return None


def build_relations(context, grid):
    """Create or refresh the relations object of ``grid``; returns it."""
    if not is_grid(grid):
        raise ValueError("Select a generated WFC grid")
    assets = find_assets(grid)
    if assets is None:
        raise ValueError("Grid has no tile collection")
    ids, width, height = grid_tiles(grid)
    data = layout(ids, width, height)

    obj = find_relations(grid)
    if obj is None:
        mesh = bpy.data.meshes.new("WFC Relations")
        obj = bpy.data.objects.new("WFC Relations", mesh)
        parent = grid.users_collection[0] if grid.users_collection else context.scene.collection
        parent.objects.link(obj)
        obj.location = (grid.location.x + width + 3.0, grid.location.y, grid.location.z)
        obj[SOURCE_PROP] = grid
        grid[OBJECT_PROP] = obj
    _write_mesh(obj.data, data)

    modifier = _modifier(obj)
    if modifier is None:
        modifier = obj.modifiers.new(MODIFIER_NAME, "NODES")
        modifier.node_group = _node_group(assets)
    else:
        modifier.node_group.nodes[COLLECTION_NODE].inputs["Collection"].default_value = assets
    obj.update_tag()
    return obj
