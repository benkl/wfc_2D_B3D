# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTIBILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

# Original WFC implementation by Maxim Gumin @mxgmn on github
# Python implementation by Victor Le @Coac on github
# Blender implementation by Benjamin Kleinert @benkl on github

bl_info = {
    "name" : "B3D Wave Function Collapse",
    "author" : "Maxim Gumin, Victor Le, Benjamin Kleinert",
    "description" : "",
    "blender" : (2, 80, 0),
    "version" : (0, 0, 1),
    "location" : "",
    "warning" : "",
    "category" : "Generic"
}

import bpy

from . wfc_panel import WFC_PT_Panel
from . wfc_2d import WFC_OT_Runner
from . wfc_panel import WFC_UI_Variables
from . wfc_moduleplacer import WFC_OT_Placer
from bpy.props import PointerProperty

classes = (
    WFC_PT_Panel,
    WFC_OT_Runner,
    WFC_UI_Variables,
    WFC_OT_Placer
)

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.wfc_vars = PointerProperty(type = WFC_UI_Variables)


def unregister():
    for c in classes:
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.wfc_vars 