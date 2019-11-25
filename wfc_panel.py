import bpy
from bpy.types import (Panel, PropertyGroup)
from bpy.props import (FloatVectorProperty, IntProperty,
                       BoolProperty, PointerProperty, StringProperty, EnumProperty)


class WFC_PT_Panel(bpy.types.Panel):
    bl_idname = "WFC_PT_Panel"
    bl_label = "Wave Function Collapse"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Misc"

    # def draw_header(self, context):
    #     layout = self.layout
    #     layout.label(text="Wave Function Collapse")

    def draw(self, context):
        layout = self.layout
        patternbox = layout.box()
        patternbox.label(text="2D Image Collapse")
        patternbox.operator(
            "image.open", text="Add Pattern Source", icon="PLUS")
        patternbox.prop(context.scene.wfc_vars, "wfc_images")
        patternrow = patternbox.row()
        patternrow.prop(context.scene.wfc_vars, "wfc_patternx")
        patternrow.prop(context.scene.wfc_vars, "wfc_patterny")
        patternbox.prop(context.scene.wfc_vars, "wfc_rot")
        patternbox.prop(context.scene.wfc_vars, "wfc_flipv")
        patternbox.prop(context.scene.wfc_vars, "wfc_fliph")
        patternbox.prop(context.scene.wfc_vars, "wfc_border")
        patternbox.prop(context.scene.wfc_vars, "wfc_borderrule")
        patternbox.prop(context.scene.wfc_vars, "wfc_resultx")
        patternbox.prop(context.scene.wfc_vars, "wfc_resulty")
        patternbox.prop(context.scene.wfc_vars, "wfc_loopcount")
        patternbox.operator("object.wfc_ot_runner", icon="PLAY")
        placerbox = layout.box()
        placerbox.label(text="2D Module Instancer")
        placerbox.operator(
            "image.open", text="Add Placer Source", icon="PLUS")
        placerbox.prop(context.scene.wfc_vars, "wfc_images_placer")
        placerbox.operator("object.wfc_ot_placer", icon="PLAY")


class WFC_UI_Variables(PropertyGroup):
    def wfc_img_list(self, context):
        return [(img.name,)*3 for img in bpy.data.images]

    wfc_images: EnumProperty(
        name="Pattern Source",
        description="Selected source pattern image, PNG strongly recommended",
        items=wfc_img_list
    )
    wfc_images_placer: EnumProperty(
        name="Placer Source",
        description="Selected pattern image to place",
        items=wfc_img_list
    )
    wfc_loopcount: IntProperty(
        name="Loop count",
        default=1,
        description="Generate multiple outputs, can take forever."
    )
    wfc_patternx: IntProperty(
        name="Pattern X",
        default=3,
        description="This defines how many neighbours are taken into account for rule creation. X-Dimension. 2-3 Recommended."
    )
    wfc_patterny: IntProperty(
        name="Pattern Y",
        default=3,
        description="This defines how many neighbours are taken into account for rule creation. Y-Dimension. 2-3 Recommended."
    )
    wfc_resultx: IntProperty(
        name="Output X",
        default=30,
        description="Output image X-Dimension, <30 recommended"
    )
    wfc_resulty: IntProperty(
        name="Output Y",
        default=30,
        description="Output image Y-Dimension, <30 recommended"
    )
    wfc_borderrule: IntProperty(
        name="Border rule index",
        default=0,
        description="This is experiemental and can run out of bounds."
    )
    wfc_fliph: BoolProperty(
        name="Flip patterns H",
        default=False,
        description="Small input images recommended, can severly prolong collapse. Adds horizontally flipped variants of all found patterns."
    )
    wfc_flipv: BoolProperty(
        name="Flip patterns V",
        default=False,
        description="Small input images recommended, can severly prolong collapse. Adds vertically flipped variants of all found patterns."
    )
    wfc_rot: BoolProperty(
        name="Rotate patterns",
        default=False,
        description="Small input images recommended, can severly prolong collapse. Adds rotated variants of all found patterns."
    )
    wfc_border: BoolProperty(
        name="Constrain grid borders",
        default=False,
        description="Create a border with the n-th rule found. Select below."
    )
