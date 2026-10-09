"""Per-grid tile outlines and editable Geometry Nodes labels."""

import bpy


COLLECTION_NODE = "WFC Appearance Collection"
COLLECTION_PROP = "wfc_appearance"
OUTLINE_MATERIAL = "WFC Outline"
FONT_MATERIAL = "WFC Tile Font"
LABEL_NODE = "WFC Label For Each Point"


def _material(name, color):
    material = bpy.data.materials.new(name)
    material.diffuse_color = tuple(color)
    material.use_nodes = True
    shader = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = tuple(color)
    shader.inputs["Roughness"].default_value = 0.8
    return material


def _set_color(material, color):
    material.diffuse_color = tuple(color)
    shader = next(n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    shader.inputs["Base Color"].default_value = tuple(color)


def _outline():
    # The stock tiles are one unit wide. The frame sits above their top surface.
    vertices, faces = [], []
    for x0, y0, x1, y1 in (
        (-0.48, -0.48, 0.48, -0.445),
        (-0.48, 0.445, 0.48, 0.48),
        (-0.48, -0.445, -0.445, 0.445),
        (0.445, -0.445, 0.48, 0.445),
    ):
        start = len(vertices)
        vertices.extend(((x0, y0, 0.185), (x1, y0, 0.185),
                         (x1, y1, 0.185), (x0, y1, 0.185)))
        faces.append((start, start + 1, start + 2, start + 3))
    mesh = bpy.data.meshes.new("WFC Tile Outline")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return mesh


def _input(group, name, socket_type, default, *, minimum=None):
    socket = group.interface.new_socket(name=name, in_out="INPUT", socket_type=socket_type)
    socket.default_value = default
    if minimum is not None:
        socket.min_value = minimum
    return socket


def _attribute(nodes, name, y):
    node = nodes.new("GeometryNodeInputNamedAttribute")
    node.data_type = "INT"
    node.inputs["Name"].default_value = name
    node.location = (-800, y)
    return node.outputs["Attribute"]


def _as_string(nodes, links, value, x, y):
    node = nodes.new("FunctionNodeValueToString")
    node.data_type = "INT"
    node.location = (x, y)
    links.new(value, node.inputs["Value"])
    return node.outputs["String"]


def _upgrade_legacy(group, collection_node, output, source):
    """Detach the old per-tile baked-label instancer, if present."""
    links, nodes = group.links, group.nodes
    old_overlay = next((link.to_node for link in collection_node.outputs["Instances"].links
                        if link.to_node.bl_idname == "GeometryNodeInstanceOnPoints"), None)
    if old_overlay is None:
        return
    old_join = next((link.to_node for link in old_overlay.outputs["Instances"].links
                     if link.to_node.bl_idname == "GeometryNodeJoinGeometry"), None)
    nodes.remove(old_overlay)
    if old_join is not None and output.inputs["Geometry"].is_linked and output.inputs["Geometry"].links[0].from_node == old_join:
        nodes.remove(old_join)
        links.new(source.outputs["Instances"], output.inputs["Geometry"])


def attach(group, collection, *, relations=False):
    """Add point-derived labels and tile outlines to a grid or relations group.

    The incoming instance node supplies positioned points with wfc_tile_id,
    wfc_source_id and wfc_variant_id attributes. Repeated calls only update
    the overlay collection and font material, preserving modifier controls.
    """
    nodes, links = group.nodes, group.links
    source = next(node for node in nodes if node.bl_idname == "GeometryNodeInstanceOnPoints"
                  and node.inputs["Points"].is_linked)
    output = next(node for node in nodes if node.bl_idname == "NodeGroupOutput")
    inp = next(node for node in nodes if node.bl_idname == "NodeGroupInput")
    collection_node = nodes.get(COLLECTION_NODE)
    if collection_node is None:
        collection_node = nodes.new("GeometryNodeCollectionInfo")
        collection_node.name = COLLECTION_NODE
        collection_node.label = "Tile Outlines"
        collection_node.transform_space = "ORIGINAL"
        collection_node.inputs["Separate Children"].default_value = True
        collection_node.inputs["Reset Children"].default_value = True
        collection_node.location = (-550, -850)
    collection_node.inputs["Collection"].default_value = collection
    font_name = collection.get("font_material")
    font = bpy.data.materials.get(font_name) if isinstance(font_name, str) else None
    if nodes.get(LABEL_NODE) is not None:
        if font is not None:
            for node in nodes:
                if node.bl_idname == "GeometryNodeSetMaterial" and node.label == "WFC Label Material":
                    node.inputs["Material"].default_value = font
        return

    _upgrade_legacy(group, collection_node, output, source)
    _input(group, "Show Labels", "NodeSocketBool", True)
    _input(group, "Label Size", "NodeSocketFloat", 0.32, minimum=0.001)
    _input(group, "Label Offset", "NodeSocketVector", (0.0, 0.0, 0.205))
    _input(group, "Label Extrude", "NodeSocketFloat", 0.002, minimum=0.0)
    _input(group, "Show Source Relation", "NodeSocketBool", True)
    # Existing Group Input nodes keep their old zero-valued sockets when sockets
    # are appended to the interface. Initialize both; the modifier still owns
    # its own values after the group is assigned.
    for name, value in (("Show Labels", True), ("Label Size", 0.32),
                        ("Label Offset", (0.0, 0.0, 0.205)),
                        ("Label Extrude", 0.002), ("Show Source Relation", True)):
        inp.outputs[name].default_value = value

    points = source.inputs["Points"].links[0].from_socket
    scale = source.inputs["Scale"].links[0].from_socket if source.inputs["Scale"].is_linked else None

    tile = _attribute(nodes, "wfc_tile_id", -1050)
    source_id = _attribute(nodes, "wfc_source_id", -1170)
    variant = _attribute(nodes, "wfc_variant_id", -1290)

    outlines = nodes.new("GeometryNodeInstanceOnPoints")
    outlines.label = "Tile Outlines"
    outlines.inputs["Pick Instance"].default_value = False
    outlines.location = (-320, -650)
    links.new(points, outlines.inputs["Points"])
    links.new(collection_node.outputs["Instances"], outlines.inputs["Instance"])
    if scale is not None:
        links.new(scale, outlines.inputs["Scale"])
    else:
        outlines.inputs["Scale"].default_value = source.inputs["Scale"].default_value

    # String fields cannot vary on the generated text geometry's domain. The
    # zone evaluates each point's integer attributes before building its text.
    each = nodes.new("GeometryNodeForeachGeometryElementInput")
    each.name = LABEL_NODE
    each.location = (-500, -1500)
    done = nodes.new("GeometryNodeForeachGeometryElementOutput")
    done.domain = "POINT"
    done.location = (1550, -1500)
    each.pair_with_output(done)
    links.new(points, each.inputs["Geometry"])
    links.new(inp.outputs["Show Labels"], each.inputs["Selection"])
    for name, field in (("Tile", tile), ("Source", source_id), ("Variant", variant)):
        done.input_items.new("INT", name)
        links.new(field, each.inputs[name])
    if scale is not None:
        done.input_items.new("VECTOR", "Tile Scale")
        links.new(scale, each.inputs["Tile Scale"])

    src_text = _as_string(nodes, links, each.outputs["Source"], -220, -1150)
    variant_text = _as_string(nodes, links, each.outputs["Variant"], -220, -1250)
    tile_text = _as_string(nodes, links, each.outputs["Tile"], -220, -1370)
    dotted = nodes.new("GeometryNodeStringJoin")
    dotted.location = (0, -1150)
    dotted.inputs["Delimiter"].default_value = "."
    links.new(src_text, dotted.inputs["Strings"])
    links.new(variant_text, dotted.inputs["Strings"])
    choice = nodes.new("GeometryNodeSwitch")
    choice.input_type = "STRING"
    choice.location = (240, -1200)
    links.new(inp.outputs["Show Source Relation"], choice.inputs["Switch"])
    links.new(tile_text, choice.inputs["False"])
    links.new(dotted.outputs["String"], choice.inputs["True"])

    text = nodes.new("GeometryNodeStringToCurves")
    text.location = (450, -1200)
    text.inputs["Align X"].default_value = "Center"
    text.inputs["Align Y"].default_value = "Middle"
    links.new(choice.outputs["Output"], text.inputs["String"])
    links.new(inp.outputs["Label Size"], text.inputs["Size"])
    realize = nodes.new("GeometryNodeRealizeInstances")
    realize.location = (650, -1200)
    links.new(text.outputs["Curve Instances"], realize.inputs["Geometry"])
    fill = nodes.new("GeometryNodeFillCurve")
    fill.location = (850, -1200)
    links.new(realize.outputs["Geometry"], fill.inputs["Curve"])
    extrude = nodes.new("GeometryNodeExtrudeMesh")
    extrude.mode = "FACES"
    extrude.inputs["Offset"].default_value = (0.0, 0.0, 1.0)
    extrude.location = (1050, -1200)
    links.new(fill.outputs["Mesh"], extrude.inputs["Mesh"])
    links.new(inp.outputs["Label Extrude"], extrude.inputs["Offset Scale"])
    material_node = nodes.new("GeometryNodeSetMaterial")
    material_node.label = "WFC Label Material"
    material_node.location = (1250, -1200)
    material_node.inputs["Material"].default_value = font
    links.new(extrude.outputs["Mesh"], material_node.inputs["Geometry"])
    offset = nodes.new("GeometryNodeTransform")
    offset.location = (1450, -1200)
    links.new(inp.outputs["Label Offset"], offset.inputs["Translation"])
    links.new(material_node.outputs["Geometry"], offset.inputs["Geometry"])
    label = nodes.new("GeometryNodeInstanceOnPoints")
    label.location = (1650, -1200)
    links.new(each.outputs["Element"], label.inputs["Points"])
    links.new(offset.outputs["Geometry"], label.inputs["Instance"])
    if scale is not None:
        links.new(each.outputs["Tile Scale"], label.inputs["Scale"])
    else:
        label.inputs["Scale"].default_value = source.inputs["Scale"].default_value
    links.new(label.outputs["Instances"], done.inputs["Geometry"])

    join = nodes.new("GeometryNodeJoinGeometry")
    join.location = (450, 0)
    previous = output.inputs["Geometry"].links[0].from_socket
    links.new(previous, join.inputs["Geometry"])
    links.new(outlines.outputs["Instances"], join.inputs["Geometry"])
    links.new(done.outputs["Generation_0"], join.inputs["Geometry"])
    links.new(join.outputs["Geometry"], output.inputs["Geometry"])
    output.location = (700, 0)


CONTROL_NAMES = ("Show Labels", "Label Size", "Label Offset", "Label Extrude",
                 "Show Source Relation")


def control_ids(group):
    """Identifiers of the label controls currently on ``group``'s interface."""
    return {item.identifier for item in group.interface.items_tree
            if item.item_type == "SOCKET" and item.in_out == "INPUT"
            and item.name in CONTROL_NAMES}


def initialize_modifier(modifier, identifiers=None):
    """Give label controls on a bound Nodes modifier their interface defaults.

    Sockets appended to a group that a modifier already uses read as zero until
    set. ``identifiers`` limits the reset to those sockets, e.g. the difference
    of ``control_ids`` before and after ``attach``; None resets every label
    control. Blender 5.3 exposes values on ``modifier.properties.inputs``; older
    builds store them as custom properties on the modifier.
    """
    group = modifier.node_group
    if group is None:
        return
    for item in group.interface.items_tree:
        if (item.item_type != "SOCKET" or item.in_out != "INPUT"
                or item.name not in CONTROL_NAMES
                or (identifiers is not None and item.identifier not in identifiers)):
            continue
        value = item.default_value
        if item.name == "Label Offset":
            value = tuple(value)
        inputs = getattr(modifier, "properties", None)
        inputs = getattr(inputs, "inputs", None)
        if inputs is not None:
            getattr(inputs, item.identifier).value = value
        else:
            modifier[item.identifier] = value
    modifier.id_data.update_tag()


def sync(context, parent, group, count, settings):
    """Refresh the grid's outline collection and its live label material."""
    node = group.nodes.get(COLLECTION_NODE)
    collection = node.inputs["Collection"].default_value if node is not None else None
    if collection is None:
        collection = bpy.data.collections.new("WFC Appearance")
        collection[COLLECTION_PROP] = True
        parent.children.link(collection)
    elif collection not in parent.children[:]:
        parent.children.link(collection)

    for key, name, color in (("outline_material", OUTLINE_MATERIAL, settings.outline_color),
                             ("font_material", FONT_MATERIAL, settings.font_color)):
        material_name = collection.get(key)
        material = bpy.data.materials.get(material_name) if isinstance(material_name, str) else None
        if material is None:
            material = _material(name, color)
            collection[key] = material.name
        else:
            _set_color(material, color)

    old = list(collection.objects)
    for obj in old:
        mesh = obj.data if obj.type == "MESH" else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    mesh = _outline()
    mesh.materials.append(bpy.data.materials[collection["outline_material"]])
    obj = bpy.data.objects.new("WFC Tile Outline", mesh)
    collection.objects.link(obj)
    attach(group, collection)
