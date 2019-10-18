import bpy
from bpy.types import (Panel, PropertyGroup)
from bpy.props import (FloatVectorProperty, FloatProperty,
                       BoolProperty, PointerProperty, StringProperty, EnumProperty)


class WFC_PT_Panel(bpy.types.Panel):
    bl_idname = "view3d.WFC_PT_Panel"
    bl_label = "Wave Function Collapse"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Misc"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Wave Function Collapse")
        layout.operator("object.wfc_ot_runner")
        layout.operator("image.open", text="Add Pattern Source", icon="PLUS")
        layout.prop(context.scene.wfc_vars, "wfc_images")


class WFC_UI_variables(PropertyGroup):
    def wfc_img_list(self, context):
        return [(img.name,)*3 for img in bpy.data.images]

    wfc_images: EnumProperty(
        name="Pattern Source",
        description="Selected source pattern image",
        items=wfc_img_list
    )
