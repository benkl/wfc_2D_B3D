"""Generate or regenerate a WFC grid from a Blender image."""

import bpy

from .wfc_geometry import build, is_grid
from .wfc_relations import SOURCE_PROP, build_relations, find_relations
from .wfc_solver import solve, tile_types


class WFC_OT_GeometryGenerate(bpy.types.Operator):
    bl_idname = "object.wfc_geometry_generate"
    bl_label = "Generate WFC Tile Grid"
    bl_description = "Solve overlapping image patterns and instance editable tiles with Geometry Nodes"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.wfc_settings
        image = settings.source
        if image is None or image.size[0] < 1 or image.size[1] < 1:
            self.report({"ERROR"}, "Choose a loaded Pattern Source image")
            return {"CANCELLED"}
        try:
            indices, colors = solve(
                image.pixels[:], image.size[0], image.size[1],
                settings.output_width, settings.output_height,
                settings.pattern_width, settings.pattern_height,
                seed=settings.seed, attempts=settings.attempts,
                rotate=settings.rotate, flip_horizontal=settings.flip_horizontal,
                flip_vertical=settings.flip_vertical,
            )
            indices, colors, keys = tile_types(
                indices, colors, settings.output_width, settings.output_height,
                settings.tile_mode)
        except (ValueError, RuntimeError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        active = context.active_object
        existing = active if is_grid(active) else None
        try:
            grid, tiles = build(context, indices, colors, keys, settings.output_width,
                                settings.output_height, existing=existing, settings=settings)
        except (ValueError, RuntimeError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        relations = find_relations(grid)
        if relations is not None:
            build_relations(context, grid)
        self.report({"INFO"}, "%s %d cells with %d tile assets" % (
            "Regenerated" if existing else "Generated", len(indices), len(colors)))
        return {"FINISHED"}



class WFC_OT_ShowRelations(bpy.types.Operator):
    bl_idname = "object.wfc_show_relations"
    bl_label = "Show Tile Relations"
    bl_description = ("Build a Geometry Nodes view that shows, for every tile, which tiles "
                      "sit on its right, upper, left and lower side in the generated grid")
    bl_options = {"REGISTER", "UNDO"}

    @staticmethod
    def _grid(obj):
        if is_grid(obj):
            return obj
        source = obj.get(SOURCE_PROP) if obj is not None else None
        return source if is_grid(source) else None

    @classmethod
    def poll(cls, context):
        return cls._grid(context.active_object) is not None

    def execute(self, context):
        grid = self._grid(context.active_object)
        try:
            view = build_relations(context, grid)
        except (ValueError, RuntimeError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        self.report({"INFO"}, "Relations view: %d points" % len(view.data.vertices))
        return {"FINISHED"}
