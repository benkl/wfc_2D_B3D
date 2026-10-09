"""Geometry Nodes WFC controls in the 3D View sidebar."""

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatVectorProperty, IntProperty, PointerProperty
from bpy.types import Panel, PropertyGroup

from .wfc_geometry import is_grid


class WFC_Settings(PropertyGroup):
    source: PointerProperty(
        name="Pattern Source", type=bpy.types.Image,
        description="Image used to learn overlapping color patterns",
    )
    pattern_width: IntProperty(name="Pattern X", default=2, min=1, max=32)
    pattern_height: IntProperty(name="Pattern Y", default=2, min=1, max=32)
    output_width: IntProperty(name="Output X", default=30, min=1, max=2048)
    output_height: IntProperty(name="Output Y", default=30, min=1, max=2048)
    rotate: BoolProperty(name="Rotate", description="Learn rotations of the source image")
    flip_horizontal: BoolProperty(name="Flip Horizontal")
    flip_vertical: BoolProperty(name="Flip Vertical")
    seed: IntProperty(name="Seed", default=0)
    attempts: IntProperty(name="Attempts", default=20, min=1, max=1000,
                          description="Retry with a new deterministic random stream on contradiction")
    tile_mode: EnumProperty(
        name="Tiles",
        description="How solved cells become editable tiles",
        items=(
            ("COLOR", "By Color", "One tile per color"),
            ("EDGES", "By Neighbors",
             "Separate tile for each color combined with its left, right, lower and upper neighbors"),
            ("EDGES_CORNERS", "By Neighbors + Corners",
             "Like By Neighbors, also including the four diagonal neighbors"),
        ),
        default="COLOR",
    )
    outline_color: FloatVectorProperty(
        name="Outline Color", subtype="COLOR", size=4, min=0.0, max=1.0,
        default=(0.07, 0.09, 0.14, 1.0),
        description="Color of the outline drawn around each tile",
    )
    font_color: FloatVectorProperty(
        name="Font Color", subtype="COLOR", size=4, min=0.0, max=1.0,
        default=(1.0, 1.0, 1.0, 1.0),
        description="Color of the tile ID numbers",
    )


class WFC_PT_Panel(Panel):
    bl_idname = "WFC_PT_panel"
    bl_label = "Wave Function Collapse"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "WFC"

    def draw(self, context):
        layout = self.layout
        active = context.active_object
        label = "Regenerate Selected Grid" if is_grid(active) else "Generate Tile Grid"
        col = layout.column(align=True)
        col.scale_y = 1.3
        col.operator("object.wfc_geometry_generate", text=label, icon="GEOMETRY_NODES")
        col.operator("object.wfc_show_relations", icon="NODETREE")
        layout.label(text="Edit tiles in the generated WFC Tiles collection")


class WFC_PT_SourcePanel(Panel):
    bl_idname = "WFC_PT_source"
    bl_label = "Source & Patterns"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "WFC"
    bl_parent_id = "WFC_PT_panel"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.wfc_settings
        layout.prop(settings, "source")
        row = layout.row(align=True)
        row.prop(settings, "pattern_width")
        row.prop(settings, "pattern_height")
        options = layout.column(align=True)
        options.prop(settings, "rotate")
        row = options.row(align=True)
        row.prop(settings, "flip_horizontal")
        row.prop(settings, "flip_vertical")
        layout.prop(settings, "tile_mode")


class WFC_PT_SolverPanel(Panel):
    bl_idname = "WFC_PT_solver"
    bl_label = "Grid & Solver"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "WFC"
    bl_parent_id = "WFC_PT_panel"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.wfc_settings
        row = layout.row(align=True)
        row.prop(settings, "output_width")
        row.prop(settings, "output_height")
        layout.prop(settings, "seed")
        layout.prop(settings, "attempts")


class WFC_PT_AppearancePanel(Panel):
    bl_idname = "WFC_PT_appearance"
    bl_label = "Tile Appearance"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "WFC"
    bl_parent_id = "WFC_PT_panel"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.wfc_settings
        layout.prop(settings, "outline_color")
        layout.prop(settings, "font_color")
