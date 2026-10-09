"""Visualise which tile neighbourhoods occur in a solved grid, driven by Geometry Nodes.

For a generated grid this builds a second object, ``WFC Relations``. Every distinct
four-side neighbourhood found in the grid gets its own block of up to five points:
the centre tile in the middle and the tile found on its right, up, left and down
side around it, each joined to the centre by a line. A side that lies outside the
grid has no point. Two neighbourhoods are the same when the centre tile and all four
neighbour tiles agree, so a repeated neighbourhood appears once and two
neighbourhoods around the same centre tile are separate blocks. Nothing is fanned
out: a block never lists alternative neighbours for one side.

Blocks are grouped by source tile (the centre colour): one row per source, one
column per neighbourhood, ordered by variant. Instances come from the grid's own
tile collection, so edits to tile meshes show up here as well. Right is +X and up is
+Y, as in the grid itself.

The mesh holds the default layout, with the centre of every block at
``(column * BLOCK_PITCH, -row * BLOCK_PITCH)`` and neighbours ``DISTANCE`` away. Point
attributes drive the node group:

* ``wfc_tile_id``    tile instanced at the point (centre tile or neighbour)
* ``wfc_source_id``  source of that tile: palette index of the centre colour
* ``wfc_variant_id`` ordinal of that tile among the tiles of its source, from 0
* ``wfc_rel_side``   -1 for the centre, else 0 right, 1 up, 2 left, 3 down
* ``wfc_rel_row``    row of the block (source group)
* ``wfc_rel_col``    column of the block within its row

The node group moves points from the default layout to the Distance and Block
Spacing inputs, shrinks neighbours, and turns the edges into tubes. Source and
variant ids are read from the grid's own ``wfc_source_id`` / ``wfc_variant_id`` point
attributes, else from the same custom properties of the tile objects, else derived
from the tile keys in tile id order. Neighbourhoods are the ones observed in the
solved grid, not every pairing the source image could allow.
"""

import math

import bpy
from .wfc_appearance import (COLLECTION_NODE as APPEARANCE_NODE, attach as attach_appearance,
                             control_ids, initialize_modifier)

from .wfc_geometry import ATTRIBUTE_NAME, COLLECTION_NODE, KEY_PROP, TILE_PROP, find_assets, is_grid

GROUP_NAME = "WFC Relations"
GROUP_PROP = "wfc_relations_group"
GROUP_VERSION = 2
VERSION_PROP = "wfc_relations_version"
OBJECT_PROP = "wfc_relations"      # grid -> relations object
SOURCE_PROP = "wfc_relations_of"   # relations object -> grid
MODIFIER_NAME = "WFC Relations"
SOURCE_ATTRIBUTE = "wfc_source_id"
VARIANT_ATTRIBUTE = "wfc_variant_id"
SIDE_ATTRIBUTE = "wfc_rel_side"
ROW_ATTRIBUTE = "wfc_rel_row"
COLUMN_ATTRIBUTE = "wfc_rel_col"

DISTANCE = 1.6   # centre tile to each neighbour
TILE_SCALE = 0.8
NEIGHBOUR_SCALE = 0.6
LINE_RADIUS = 0.02
_BLOCK_MARGIN = 0.8
BLOCK_PITCH = 2.0 * (DISTANCE + _BLOCK_MARGIN) + 0.4   # distance between blocks
# (dx, dy) of each side: 0 right, 1 up, 2 left, 3 down.
_SIDE_OFFSETS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def find_relations(grid):
    """The relations object of a grid, or None."""
    obj = grid.get(OBJECT_PROP)
    if isinstance(obj, bpy.types.Object) and obj.get(SOURCE_PROP) == grid:
        return obj
    return None


def relations_modifier(obj):
    """The relations Geometry Nodes modifier of ``obj``, or None."""
    for modifier in obj.modifiers:
        group = getattr(modifier, "node_group", None)
        if modifier.type == "NODES" and group is not None and group.get(GROUP_PROP):
            return modifier
    return None


def _int_attribute(mesh, name):
    """Values of an integer point attribute, or None when it is absent."""
    attribute = mesh.attributes.get(name)
    if attribute is None or attribute.domain != "POINT" or attribute.data_type != "INT":
        return None
    values = [0] * len(mesh.vertices)
    attribute.data.foreach_get("value", values)
    return values


def grid_tiles(grid):
    """Tile ids, size, and optional source/variant ids of a generated grid object.

    Returns ``(ids, width, height, sources, variants)``; the last two are lists
    parallel to ``ids``, or None when the grid does not carry them.
    """
    mesh = grid.data
    count = len(mesh.vertices)
    ids = _int_attribute(mesh, ATTRIBUTE_NAME)
    if ids is None or count == 0:
        raise ValueError("Grid has no wfc_tile_id point attribute")
    coordinates = [0.0] * (count * 3)
    mesh.vertices.foreach_get("co", coordinates)
    width = int(round(max(coordinates[0::3]))) + 1
    if count % width:
        raise ValueError("Grid points do not form a rectangle")
    return (ids, width, count // width,
            _int_attribute(mesh, SOURCE_ATTRIBUTE), _int_attribute(mesh, VARIANT_ATTRIBUTE))


def tile_identities(ids, assets, sources=None, variants=None):
    """Map each tile id in ``ids`` to ``(source, variant)``.

    Uses the grid point attributes ``sources`` / ``variants`` when given, else the
    ``wfc_source_id`` / ``wfc_variant_id`` properties of the tile objects, else
    groups the tiles by the centre colour of their key in tile id order: the source
    is the group's rank and the variant the ordinal inside the group.
    """
    if sources is not None and variants is not None:
        return {tile: (sources[i], variants[i]) for i, tile in reversed(list(enumerate(ids)))}

    objects = {}
    for obj in assets.objects:
        tile = obj.get(TILE_PROP)
        if isinstance(tile, int):
            objects[tile] = obj
    wanted = sorted(set(ids))
    missing = [t for t in wanted if t not in objects]
    if missing:
        raise ValueError("Tile collection has no object for tile id %d" % missing[0])

    result = {}
    for tile in wanted:
        source = objects[tile].get(SOURCE_ATTRIBUTE)
        variant = objects[tile].get(VARIANT_ATTRIBUTE)
        if not isinstance(source, int) or not isinstance(variant, int):
            result = None
            break
        result[tile] = (source, variant)
    if result is not None:
        return result

    # Derive from keys: the centre colour is the part before the first "|".
    result = {}
    rank = {}
    seen = {}
    for tile in sorted(objects):
        key = objects[tile].get(KEY_PROP)
        if not isinstance(key, str):
            raise ValueError("Tile %d has no wfc_tile_key" % tile)
        centre = key.split("|", 1)[0]
        source = rank.setdefault(centre, len(rank))
        variant = seen.get(source, 0)
        seen[source] = variant + 1
        result[tile] = (source, variant)
    return {tile: result[tile] for tile in wanted}


def configurations(tile_indices, width, height):
    """Distinct four-side neighbourhoods of a grid of tile ids.

    Returns a sorted list of ``(centre, right, up, left, down)``; a neighbour is -1
    when that side lies outside the grid. A neighbourhood seen several times is
    listed once.
    """
    if len(tile_indices) != width * height:
        raise ValueError("expected %d tile indices, got %d" % (width * height, len(tile_indices)))
    found = set()
    for y in range(height):
        for x in range(width):
            around = [tile_indices[y * width + x]]
            for dx, dy in _SIDE_OFFSETS:
                nx, ny = x + dx, y + dy
                inside = 0 <= nx < width and 0 <= ny < height
                around.append(tile_indices[ny * width + nx] if inside else -1)
            found.add(tuple(around))
    return sorted(found)


def layout(tile_indices, width, height, identity):
    """Mesh data for the relations view of a solved grid.

    ``identity`` maps every tile id of the grid to ``(source, variant)``. Returns
    ``(positions, edges, attributes)`` where ``attributes`` maps each point
    attribute name to one integer per vertex. Blocks are ordered by source, then by
    centre tile (variant), then by neighbours; the centre vertex of a block is
    followed by its present neighbours in the order right, up, left, down.
    """
    blocks = sorted(configurations(tile_indices, width, height),
                    key=lambda c: (identity[c[0]], c[0], c[1:]))
    rows = {}
    for block in blocks:
        rows.setdefault(identity[block[0]][0], []).append(block)

    positions = []
    edges = []
    names = (ATTRIBUTE_NAME, SOURCE_ATTRIBUTE, VARIANT_ATTRIBUTE,
             SIDE_ATTRIBUTE, ROW_ATTRIBUTE, COLUMN_ATTRIBUTE)
    attributes = {name: [] for name in names}

    def point(position, tile, side, row, column):
        source, variant = identity[tile]
        positions.append(position)
        for name, value in zip(names, (tile, source, variant, side, row, column)):
            attributes[name].append(value)
        return len(positions) - 1

    for row, source in enumerate(sorted(rows)):
        for column, block in enumerate(rows[source]):
            origin = (column * BLOCK_PITCH, -row * BLOCK_PITCH, 0.0)
            centre = point(origin, block[0], -1, row, column)
            for side, neighbour in enumerate(block[1:]):
                if neighbour < 0:
                    continue
                dx, dy = _SIDE_OFFSETS[side]
                edges.append((centre, point((origin[0] + dx * DISTANCE, origin[1] + dy * DISTANCE, 0.0),
                                            neighbour, side, row, column)))
    return positions, edges, attributes


def _write_mesh(mesh, data):
    positions, edges, attributes = data
    mesh.clear_geometry()
    mesh.from_pydata(positions, edges, [])
    mesh.update()
    for name in [a.name for a in mesh.attributes if a.name.startswith("wfc_")]:
        if name not in attributes:
            mesh.attributes.remove(mesh.attributes[name])
    for name, values in attributes.items():
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
    group[VERSION_PROP] = GROUP_VERSION
    group.interface.new_socket(name="Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    _socket(group, "Tile Scale", TILE_SCALE, 0.01)
    _socket(group, "Neighbour Scale", NEIGHBOUR_SCALE, 0.01)
    _socket(group, "Distance", DISTANCE, 0.0)
    _socket(group, "Block Spacing", BLOCK_PITCH, 0.0)
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
    row = attribute(ROW_ATTRIBUTE, -460)
    column = attribute(COLUMN_ATTRIBUTE, -590)

    # The mesh holds the default layout; Distance and Block Spacing move points away
    # from it. The centre tile has side -1: it keeps full size and is not pushed out.
    has_side = _math(group, "GREATER_THAN", side, -0.5, x=-1250, y=-300)
    angle = _math(group, "MULTIPLY", side, math.pi / 2.0, x=-1250, y=-420)
    cos_a = _math(group, "COSINE", angle, x=-1050, y=-380)
    sin_a = _math(group, "SINE", angle, x=-1050, y=-480)
    push = _math(group, "SUBTRACT", inp.outputs["Distance"], DISTANCE, x=-1250, y=-180)
    radial = _math(group, "MULTIPLY", has_side, push, x=-1050, y=-250)
    stretch = _math(group, "SUBTRACT", inp.outputs["Block Spacing"], BLOCK_PITCH, x=-1250, y=-680)
    x_off = _math(group, "ADD",
                  _math(group, "MULTIPLY", cos_a, radial, x=-850, y=-300),
                  _math(group, "MULTIPLY", column, stretch, x=-850, y=-500), x=-650, y=-400)
    y_off = _math(group, "SUBTRACT",
                  _math(group, "MULTIPLY", sin_a, radial, x=-850, y=-600),
                  _math(group, "MULTIPLY", row, stretch, x=-850, y=-700), x=-650, y=-650)
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


def build_relations(context, grid):
    """Create or refresh the relations object of ``grid``; returns it.

    A modifier whose node group predates the per-configuration layout gets a fresh
    group; a current group is kept so its settings and added nodes survive.
    """
    if not is_grid(grid):
        raise ValueError("Select a generated WFC grid")
    assets = find_assets(grid)
    if assets is None:
        raise ValueError("Grid has no tile collection")
    ids, width, height, sources, variants = grid_tiles(grid)
    identity = tile_identities(ids, assets, sources, variants)
    data = layout(ids, width, height, identity)

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

    grid_group = next(mod.node_group for mod in grid.modifiers
                      if mod.type == "NODES" and mod.node_group.get("wfc_node_group"))
    appearance = grid_group.nodes.get(APPEARANCE_NODE)
    collection = appearance.inputs["Collection"].default_value if appearance is not None else None
    modifier = relations_modifier(obj)
    if modifier is None or modifier.node_group.get(VERSION_PROP) != GROUP_VERSION:
        group = _node_group(assets)
        if collection is not None:
            attach_appearance(group, collection, relations=True)
        if modifier is None:
            modifier = obj.modifiers.new(MODIFIER_NAME, "NODES")
        else:
            old = modifier.node_group
            modifier.node_group = None
            if old.users == 0:
                bpy.data.node_groups.remove(old)
        modifier.node_group = group
    else:
        modifier.node_group.nodes[COLLECTION_NODE].inputs["Collection"].default_value = assets
        if collection is not None:
            before = control_ids(modifier.node_group)
            attach_appearance(modifier.node_group, collection, relations=True)
            initialize_modifier(modifier, control_ids(modifier.node_group) - before)
    obj.update_tag()
    return obj
