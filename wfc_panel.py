import bpy
from bpy.types import Panel


class WFC_PT_Panel(bpy.types.Panel):
    bl_idname = "view3d.WFC_PT_Panel"
    bl_label = "Wave Function Collapse"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Misc"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Wave Function Collapse")
        layout.operator('object.wfc_ot_runner')
