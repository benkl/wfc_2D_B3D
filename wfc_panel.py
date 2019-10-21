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
        patternbox.label(text="Rule settings")
        patternbox.operator(
            "image.open", text="Add Pattern Source", icon="PLUS")
        patternbox.prop(context.scene.wfc_vars, "wfc_images")
        patternbox.prop(context.scene.wfc_vars, "wfc_patternx")
        patternbox.prop(context.scene.wfc_vars, "wfc_patterny")
        outputbox = layout.box()
        outputbox.label(text="Output dimensions")
        outputbox.prop(context.scene.wfc_vars, "wfc_resultx")
        outputbox.prop(context.scene.wfc_vars, "wfc_resulty")
        layout.operator("object.wfc_ot_runner")


class WFC_UI_variables(PropertyGroup):
    def wfc_img_list(self, context):
        return [(img.name,)*3 for img in bpy.data.images]

    wfc_images: EnumProperty(
        name="Pattern Source",
        description="Selected source pattern image",
        items=wfc_img_list
    )

    wfc_patternx: IntProperty(
        name="Pattern X",
        default=2,
        description="This defines how many neighbours are taken into account for rule creation. X-Dimension. 2-3 Recommended."
    )
    wfc_patterny: IntProperty(
        name="Pattern Y",
        default=2,
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
