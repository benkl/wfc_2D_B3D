This is an experimental add-on for Blender implementing 2D Wave Function Collapse. For now enjoy exploring. (wfc explanation and original concept: https://github.com/mxgmn/WaveFunctionCollapse)

The wfc panel is located in the n-panel under ‘misc’.

Add a Pattern source. [png, small size recommended]
In the Pattern Source dropdown select the image you added.
Pattern X and Pattern Y are used to set the pattern rule width and height. [2 * 2, 3 * 3 recommended]
The rotation and flipping checkboxes can be set to transform the created rules [eg 2 * 2 rules from a png], rotating them or flipping them.
Constrain grid borders is used to force the first pattern being detected to be used around the grid borders. Leaving a uniform spot in the top right of your png source can serve to create a blank rule, generating grid contained patterns.
Border rule index box can be used to set the rule selected to create the grid borders mentioned above. [This is an experimental feature]
Output X and Y are used to set the size of the image you want to generate.
Loop count can be used to loop the whole operation. Should be used with caution.
Press Collapse to create an output image data block. Open the image viewer to browse through the “MyImage” results once it has completed and save them externally.

The module instancer is a seperate operator that can take any input image and create a set of seperate instances for every colour present in the image and it’s neighbouring colours. Adding a placer source allows you to load an external image, if you dont have an image from the previous generation step (MyImage e.g.). Select it in the dropdown and hit place to get a set of linked planes that can be modified to the desired geometry (or by linking your meshes). Hitting place with a different image of the same generation process will reuse existing tiles.

Toggle the system console to see progress.

Please feel free to improve the code and commit!

![Interface](https://blenderartists.org/uploads/default/original/4X/8/6/d/86dff233e6cc57d7a82effabe202ab51f4d9c896.png)

![Example Setup](https://blenderartists.org/uploads/default/original/4X/a/2/e/a2ebf70813519d612b611be6e4da00e2b150f433.png)
