"""Image-sampled WFC tile generation with editable Geometry Nodes instances."""

bl_info = {
    "name": "WFC Tile Grid",
    "author": "Benjamin Kleinert and contributors",
    "version": (1, 1, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > WFC",
    "description": "Overlapping image WFC with editable Geometry Nodes tile instances",
    "category": "Object",
}

import bpy
from bpy.props import PointerProperty

from .wfc_geometry_operator import WFC_OT_GeometryGenerate, WFC_OT_ShowRelations
from .wfc_panel import (
    WFC_PT_AppearancePanel,
    WFC_PT_Panel,
    WFC_PT_SolverPanel,
    WFC_PT_SourcePanel,
    WFC_Settings,
)


_CLASSES = (
    WFC_Settings,
    WFC_OT_GeometryGenerate,
    WFC_OT_ShowRelations,
    WFC_PT_Panel,
    WFC_PT_SourcePanel,
    WFC_PT_SolverPanel,
    WFC_PT_AppearancePanel,
)


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.wfc_settings = PointerProperty(type=WFC_Settings)


def unregister():
    del bpy.types.Scene.wfc_settings
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
