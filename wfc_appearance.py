"""Generated, per-grid tile outlines and numeric labels.

Appearance lives in a second instance collection so edits to source tile meshes,
materials and transforms survive regeneration.
"""

import bpy


COLLECTION_NODE = "WFC Appearance Collection"
COLLECTION_PROP = "wfc_appearance"
OUTLINE_MATERIAL = "WFC Outline"
FONT_MATERIAL = "WFC Tile Font"


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


def _outline(vertices, faces, materials):
    # A thin frame on the top of the stock 1x1 tile; no surfaces overlap the label.
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
        materials.append(0)


def _label(context, collection, number, vertices, faces, materials):
    curve = bpy.data.curves.new("WFC Label", "FONT")
    curve.body = str(number)
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    curve.extrude = 0.001
    obj = bpy.data.objects.new("WFC Label", curve)
    collection.objects.link(obj)
    try:
        context.view_layer.update()
        evaluated = obj.evaluated_get(context.evaluated_depsgraph_get())
        text_mesh = bpy.data.meshes.new_from_object(evaluated)
        try:
            if text_mesh.vertices:
                xs = [vertex.co.x for vertex in text_mesh.vertices]
                ys = [vertex.co.y for vertex in text_mesh.vertices]
                width = max(xs) - min(xs)
                height = max(ys) - min(ys)
                scale = min(0.7 / width if width else 1.0,
                            0.35 / height if height else 1.0)
                cx = (max(xs) + min(xs)) * 0.5
                cy = (max(ys) + min(ys)) * 0.5
                start = len(vertices)
                vertices.extend(((v.co.x - cx) * scale, (v.co.y - cy) * scale,
                                 0.205 + v.co.z * scale) for v in text_mesh.vertices)
                for polygon in text_mesh.polygons:
                    faces.append(tuple(start + i for i in polygon.vertices))
                    materials.append(1)
        finally:
            bpy.data.meshes.remove(text_mesh)
    finally:
        bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.curves.remove(curve)


def _mesh(context, collection, number, outline, font):
    vertices, faces, material_indices = [], [], []
    _outline(vertices, faces, material_indices)
    _label(context, collection, number, vertices, faces, material_indices)
    mesh = bpy.data.meshes.new("WFC Appearance %03d" % number)
    mesh.from_pydata(vertices, [], faces)
    mesh.materials.append(outline)
    mesh.materials.append(font)
    for polygon, index in zip(mesh.polygons, material_indices):
        polygon.material_index = index
    mesh.update()
    return mesh


def sync(context, parent, group, count, settings):
    """Create/update one generated overlay per active tile, in tile-ID order."""
    node = group.nodes.get(COLLECTION_NODE)
    collection = node.inputs["Collection"].default_value if node is not None else None
    if collection is None:
        collection = bpy.data.collections.new("WFC Appearance")
        collection[COLLECTION_PROP] = True
        parent.children.link(collection)
    elif collection not in parent.children[:]:
        # A moved grid may have brought its assets but not its appearance collection.
        parent.children.link(collection)

    outline = collection.get("outline_material")
    font = collection.get("font_material")
    outline = bpy.data.materials.get(outline) if isinstance(outline, str) else None
    font = bpy.data.materials.get(font) if isinstance(font, str) else None
    if outline is None:
        outline = _material(OUTLINE_MATERIAL, settings.outline_color)
        collection["outline_material"] = outline.name
    else:
        _set_color(outline, settings.outline_color)
    if font is None:
        font = _material(FONT_MATERIAL, settings.font_color)
        collection["font_material"] = font.name
    else:
        _set_color(font, settings.font_color)

    old = list(collection.objects)
    for obj in old:
        mesh = obj.data if obj.type == "MESH" else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for number in range(count):
        obj = bpy.data.objects.new("WFC Label %03d" % number,
                                   _mesh(context, collection, number, outline, font))
        collection.objects.link(obj)
    if node is None:
        _attach(group, collection)
    else:
        node.inputs["Collection"].default_value = collection


def _attach(group, collection):
    nodes, links = group.nodes, group.links
    source = next(node for node in nodes if node.bl_idname == "GeometryNodeInstanceOnPoints")
    output = next(node for node in nodes if node.bl_idname == "NodeGroupOutput")
    input_node = next(node for node in nodes if node.bl_idname == "NodeGroupInput")
    tile_id = next(node for node in nodes if node.bl_idname == "GeometryNodeInputNamedAttribute"
                   and node.inputs["Name"].default_value == "wfc_tile_id")
    collection_node = nodes.new("GeometryNodeCollectionInfo")
    collection_node.name = COLLECTION_NODE
    collection_node.label = "Tile Outlines and Labels"
    collection_node.transform_space = "ORIGINAL"
    collection_node.inputs["Separate Children"].default_value = True
    collection_node.inputs["Reset Children"].default_value = True
    collection_node.inputs["Collection"].default_value = collection
    collection_node.location = (-350, -580)
    overlay = nodes.new("GeometryNodeInstanceOnPoints")
    overlay.inputs["Pick Instance"].default_value = True
    overlay.location = (-80, -470)
    links.new(input_node.outputs["Geometry"], overlay.inputs["Points"])
    links.new(collection_node.outputs["Instances"], overlay.inputs["Instance"])
    links.new(tile_id.outputs["Attribute"], overlay.inputs["Instance Index"])
    links.new(source.inputs["Scale"].links[0].from_socket, overlay.inputs["Scale"])
    join = nodes.new("GeometryNodeJoinGeometry")
    join.location = (180, 0)
    output.location = (420, 0)
    links.new(source.outputs["Instances"], join.inputs["Geometry"])
    links.new(overlay.outputs["Instances"], join.inputs["Geometry"])
    links.new(join.outputs["Geometry"], output.inputs["Geometry"])
