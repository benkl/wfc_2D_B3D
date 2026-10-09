# WFC Tile Grid

A Blender 4.5+ extension that learns overlapping color patterns from an image, solves a 2D grid with Wave Function Collapse, and instances editable tiles using Geometry Nodes. The collapse runs in Python; Geometry Nodes handles the resulting geometry. No external Python packages or services are needed.

## Install

From the add-on directory, build an installable extension ZIP with Blender:

```sh
mkdir -p dist
blender --background --command extension build --source-dir . --output-dir dist
```

In Blender: **Edit → Preferences → Get Extensions → Install from Disk**, select `dist/wfc_tile_grid-1.1.0.zip`, and enable it. Alternatively, install this source directory as an add-on through your Blender scripts path. Restart Blender after replacing an installed version.

## Generate

1. Load a small source image with **Image → Open** in Blender's Image Editor. In the 3D View sidebar, open the **WFC** tab and select it as **Pattern Source**.
2. Set **Pattern X/Y** (2×2 or 3×3 is a good start), **Output X/Y**, and optional rotation/flips. The source image must be at least as large as the pattern in each dimension.
3. Set **Seed** for a repeatable result and **Attempts** for retries after contradictions.
4. Choose **Tiles**: *By Color* makes one tile per color. *By Neighbors* makes a separate tile for each color combined with the colors of its left, right, lower and upper neighbors, so edges, ends and junctions get their own tile to model. *By Neighbors + Corners* also includes the four diagonals. Cells on the output border count as having no neighbor on that side. Each tile type is a mesh in the tile collection; the count can grow quickly, and more than 256 tile types is rejected. Click **Generate Tile Grid**.
5. The selected `WFC Grid` object has a Geometry Nodes modifier and an integer point attribute named `wfc_tile_id`. The generated `WFC Tiles` collection holds source meshes named `Tile 000`, `Tile 001`, etc. Edit or replace their mesh data to change the instanced geometry. Each tile stores its identity in the `wfc_tile_key` custom property. **Tile Scale** in the modifier changes tile size; grid spacing remains one Blender unit.
6. Select the generated grid and click **Regenerate Selected Grid** to solve again using current settings while retaining editable tile assets. Edited tiles are reused when their `wfc_tile_key` appears again. Tiles that are not used by the current result are kept at the end of the collection for future runs, including when you switch **Tiles** mode. Deselect the grid to create a separate output.
7. Click **Show Tile Relations** with the grid selected to see which tiles touch which. This adds a `WFC Relations` object with its own Geometry Nodes modifier. Each tile in the grid gets a block. The tile sits in the middle, and the tiles found next to it in the grid are placed on the matching side (right is +X, up is +Y) and joined to it by a line. The neighbors are instances of your own tile meshes, so edits to them show up here too. Relations come from the solved grid, so a pairing the source image allows but the grid never produced is not drawn. **Neighbour Scale**, **Distance**, **Spacing** and **Line Radius** in the modifier change the layout. Regenerating the grid refreshes the view.

The sample wraps at its edges when learning rules; the output does not wrap. Color channels are compared at four decimal places. Output cells are colored by the origin of each solved overlapping pattern. The palette index is an instance index, not a material selection attribute. A higher pattern size or a noisy image can create many unique rules and make solving slow or contradictory. Try a smaller, cleaner source or a different seed if every attempt fails.

This extension is a clean cutover from the experimental Blender 2.80 add-on: the old image-only **Collapse** and neighborhood-plane **Module Instancer** operators are not packaged. Existing generated objects are ordinary Blender data and remain in saved scenes; old scene settings are not migrated.

Original WFC concept: [Maxim Gumin's WaveFunctionCollapse](https://github.com/mxgmn/WaveFunctionCollapse). Original Blender implementation: Benjamin Kleinert, with inspiration from Victor Le's Python implementation. License: MIT.
