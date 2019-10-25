################################
# WAVE FUNCTION COLLAPSE IN 2D #
################################

# Original WFC implementation by Maxim Gumin @mxgmn on github
# Python implementation by Victor Le @Coac on github
# Blender implementation by Benjamin Kleinert @benkl on github

import time
import os
import numpy as np
import random
import sys
import bpy


class WaveFunctionCollapse:

    # WaveFunctionCollapse encapsulates the wfc algorithm

    def __init__(self, grid_size, sample, pattern_size):
        self.patterns = Pattern.from_sample(sample, pattern_size)
        self.grid = self._create_grid(grid_size)
        self.propagator = Propagator(self.patterns)

    def run(self):
        start_time = time.time()

        done = False
        while not done:
            done = self.step()

        print("WFC run took %s seconds" % (time.time() - start_time))

    def step(self):
        self.grid.print_allowed_pattern_count()
        cell = self.observe()
        if cell is None:
            return True
        self.propagate(cell)
        return False

    def get_image(self):
        return self.grid.get_image()

    def get_patterns(self):
        return [pattern.to_image() for pattern in self.patterns]

    def observe(self):
        if self.grid.check_contradiction():
            return None
        cell = self.grid.find_lowest_entropy()

        if cell is None:
            return None

        cell.choose_rnd_pattern()

        return cell

    def propagate(self, cell):
        self.propagator.propagate(cell)

    def _create_grid(self, grid_size):
        num_pattern = len(self.patterns)
        return Grid(grid_size, num_pattern)


class Grid:

    # Grid is made of Cells

    def __init__(self, size, num_pattern):
        self.size = size
        self.grid = np.empty(self.size, dtype=object)
        for position in np.ndindex(self.size):
            self.grid[position] = Cell(num_pattern, position, self)

        # self.grid = np.array([[Cell(num_pattern, (x, y), self) for x in range(self.size)] for y in range(self.size)])
        # self.grid = np.array([Cell(num_pattern, (x,), self) for x in range(self.size)])

    def find_lowest_entropy(self):
        min_entropy = 999999
        lowest_entropy_cells = []
        for cell in self.grid.flat:
            if cell.is_stable():
                continue

            entropy = cell.entropy()

            if entropy == min_entropy:
                lowest_entropy_cells.append(cell)
            elif entropy < min_entropy:
                min_entropy = entropy
                lowest_entropy_cells = [cell]

        if len(lowest_entropy_cells) == 0:
            return None
        cell = lowest_entropy_cells[np.random.randint(
            len(lowest_entropy_cells))]
        return cell

    def get_cell(self, index):

        # Returns the cell contained in the grid at the provided index
        # :param index: (...z, y, x)
        # :return: cell

        return self.grid[index]

    def get_image(self):

        # Returns the grid converted from index to back to color
        # :return:

        image = np.vectorize(lambda c: c.get_value())(self.grid)
        image = Pattern.index_to_img(image)
        return image

    def check_contradiction(self):
        for cell in self.grid.flat:
            if len(cell.allowed_patterns) == 0:
                return True
        return False

    def print_allowed_pattern_count(self):
        grid_allowed_patterns = np.vectorize(
            lambda c: len(c.allowed_patterns))(self.grid)
        print(grid_allowed_patterns)


class Propagator:

    #  Propagator that computes and stores the legal patterns relative to another

    def __init__(self, patterns):
        self.patterns = patterns
        self.offsets = [(z, y, x) for x in range(-1, 2)
                        for y in range(-1, 2) for z in range(-1, 2)]

        start_time = time.time()
        self.precompute_legal_patterns()
        print("Patterns constraints generation took %s seconds" %
              (time.time() - start_time))

    def precompute_legal_patterns(self):
        # pool = Pool(os.cpu_count())
        # pool = Pool(1)

        patterns_offsets = []
        # patterns_var = []
        # offsets_var = []
        for pattern in self.patterns:
            # patterns_var.append(pattern[0][0])
            for offset in self.offsets:
                patterns_offsets.append((pattern, offset))
                # offsets_var.append(pattern[0][1])

        # patterns_compatibility = pool.starmap(
        #     self.legal_patterns, patterns_offsets)
        # pool.close()
        # pool.join()
        patterns_compatibility = []
        for i, pattern in enumerate(patterns_offsets):
            patterns_compatibility.append(self.legal_patterns(
                patterns_offsets[i][0], patterns_offsets[i][1]))

        # patterns_compatibility = self.legal_patterns(patterns_var, offsets_var)

        for pattern_index, offset, legal_patterns in patterns_compatibility:
            self.patterns[pattern_index].set_legal_patterns(
                offset, legal_patterns)

    def legal_patterns(self, pattern, offset):
        legal_patt = []
        for candidate_pattern in self.patterns:
            if pattern.is_compatible(candidate_pattern, offset):
                legal_patt.append(candidate_pattern.index)
        pattern.set_legal_patterns(offset, legal_patt)

        return pattern.index, offset, legal_patt

    @staticmethod
    def propagate(cell):
        to_update = [neighbour for neighbour, _ in cell.get_neighbors()]
        while len(to_update) > 0:
            cell = to_update.pop(0)
            for neighbour, offset in cell.get_neighbors():
                for pattern_index in cell.allowed_patterns:
                    pattern = Pattern.from_index(pattern_index)
                    pattern_still_compatible = False
                    for neighbour_pattern_index in neighbour.allowed_patterns:
                        neighbour_pattern = Pattern.from_index(
                            neighbour_pattern_index)

                        if pattern.is_compatible(neighbour_pattern, offset):
                            pattern_still_compatible = True
                            break

                    if not pattern_still_compatible:
                        cell.allowed_patterns.remove(pattern_index)

                        for neigh, _ in cell.get_neighbors():
                            if neigh not in to_update:
                                to_update.append(neigh)


class Pattern:

    # Pattern is a configuration of tiles from the input image.

    index_to_pattern = {}
    color_to_index = {}
    index_to_color = {}

    def __init__(self, data, index):
        self.index = index
        self.data = np.array(data)
        self.legal_patterns_index = {}  # offset -> [pattern_index]

    def get(self, index=None):
        if index is None:
            return self.data.item(0)
        return self.data[index]

    def set_legal_patterns(self, offset, legal_patterns):
        self.legal_patterns_index[offset] = legal_patterns

    @property
    def shape(self):
        return self.data.shape

    def is_compatible(self, candidate_pattern, offset):

        # Check if pattern is compatible with a candidate pattern for a given offset
        # :param candidate_pattern:
        # :param offset:
        # :return: True if compatible

        assert (self.shape == candidate_pattern.shape)

        # Precomputed compatibility
        if offset in self.legal_patterns_index:
            return candidate_pattern.index in self.legal_patterns_index[offset]

        # Computing compatibility
        ok_constraint = True
        start = tuple([max(offset[i], 0) for i, _ in enumerate(offset)])
        end = tuple([min(self.shape[i] + offset[i], self.shape[i])
                     for i, _ in enumerate(offset)])
        for index in np.ndindex(end):  # index = (x, y, z...)
            start_constraint = True
            for i, d in enumerate(index):
                if d < start[i]:
                    start_constraint = False
                    break
            if not start_constraint:
                continue

            if candidate_pattern.get(tuple(np.array(index) - np.array(offset))) != self.get(index):
                ok_constraint = False
                break

        return ok_constraint

    def to_image(self):
        return Pattern.index_to_img(self.data)

    @staticmethod
    def from_sample(sample, pattern_size):

        # Compute patterns from sample
        # :param pattern_size:
        # :param sample:
        # :return: list of patterns

        sample = Pattern.sample_img_to_indexes(sample)

        shape = sample.shape
        patterns = []
        pattern_index = 0

        for index, _ in np.ndenumerate(sample):
            # Checking if index is out of bounds
            out = False
            for i, d in enumerate(index):  # d is a dimension, e.g.: x, y, z
                if d > shape[i] - pattern_size[i]:
                    out = True
                    break
            if out:
                continue

            pattern_location = [range(d, pattern_size[i] + d)
                                for i, d in enumerate(index)]
            pattern_data = sample[np.ix_(*pattern_location)]
            flipdata = bpy.context.scene.wfc_vars.wfc_rotflip
            datas = [pattern_data, np.fliplr(pattern_data)]
            if shape[1] > 1 and flipdata == True:  # is 2D
                datas.append(np.flipud(pattern_data))
                datas.append(np.rot90(pattern_data, axes=(1, 2)))
                datas.append(np.rot90(pattern_data, 2, axes=(1, 2)))
                datas.append(np.rot90(pattern_data, 3, axes=(1, 2)))

            if shape[0] > 1 and flipdata == True:  # is 3D
                datas.append(np.flipud(pattern_data))
                datas.append(np.rot90(pattern_data, axes=(0, 2)))
                datas.append(np.rot90(pattern_data, 2, axes=(0, 2)))
                datas.append(np.rot90(pattern_data, 3, axes=(0, 2)))

            if shape[1] > 1 and flipdata == False:  # is 2D
                datas.append(pattern_data)

            if shape[0] > 1 and flipdata == False:  # is 3D
                datas.append(pattern_data)

            # Checking existence
            # TODO: more probability to multiple occurrences when observe phase
            for data in datas:
                exist = False
                for p in patterns:
                    if (p.data == data).all():
                        exist = True
                        break
                if exist:
                    continue

                pattern = Pattern(data, pattern_index)
                patterns.append(pattern)
                Pattern.index_to_pattern[pattern_index] = pattern
                pattern_index += 1

        # Pattern.plot_patterns(patterns)
        return patterns

    @staticmethod
    def sample_img_to_indexes(sample):

        # Convert a rgb image to a 2D array with pixel index
        # :param sample:
        # :return: pixel index sample

        Pattern.color_to_index = {}
        Pattern.index_to_color = {}
        sample_index = np.zeros(sample.shape[:-1])  # without last rgb dim
        color_number = 0
        for index in np.ndindex(sample.shape[:-1]):
            color = tuple(sample[index])
            if color not in Pattern.color_to_index:
                Pattern.color_to_index[color] = color_number
                Pattern.index_to_color[color_number] = color
                color_number += 1

            sample_index[index] = Pattern.color_to_index[color]

        print('Unique color count = ', color_number)
        return sample_index

    @staticmethod
    def index_to_img(sample):
        color = next(iter(Pattern.index_to_color.values()))

        image = np.zeros(sample.shape + (len(color),))
        for index in np.ndindex(sample.shape):
            pattern_index = sample[index]
            if pattern_index == -1:
                image[index] = [0.5 for _ in range(len(color))]  # Grey
            else:
                image[index] = Pattern.index_to_color[pattern_index]
        return image

    @staticmethod
    def from_index(pattern_index):
        return Pattern.index_to_pattern[pattern_index]


class Cell:

    # Cell is a pixel or tile (in 2d) that stores the possible patterns

    def __init__(self, num_pattern, position, grid):
        self.num_pattern = num_pattern
        self.allowed_patterns = [i for i in range(self.num_pattern)]

        self.position = position
        self.grid = grid
        self.offsets = [(z, y, x) for x in range(-1, 2)
                        for y in range(-1, 2) for z in range(-1, 2)]

    def entropy(self):
        return len(self.allowed_patterns)

    def choose_rnd_pattern(self):
        chosen_index = np.random.randint(len(self.allowed_patterns))
        self.allowed_patterns = [self.allowed_patterns[chosen_index]]

    def is_stable(self):
        return len(self.allowed_patterns) == 1

    def get_value(self):
        if self.is_stable():
            pattern = Pattern.from_index(self.allowed_patterns[0])
            return pattern.get()
        return -1

    def get_neighbors(self):
        neighbors = []
        for offset in self.offsets:
            neighbor_pos = tuple(np.array(self.position) + np.array(offset))
            out = False
            for i, d in enumerate(neighbor_pos):
                if not 0 <= d < self.grid.size[i]:
                    out = True
            if out:
                continue

            neighbors.append((self.grid.get_cell(neighbor_pos), offset))

        return neighbors


def load_sample(path):

    sample = path

    # Expand dimensions from 2D to 3D (For use in 2D)
    sample = np.expand_dims(sample, axis=0)
    sample = sample[:, :, :, :3]

    return sample


###########################
# WELCOME TO BLENDER CITY #
###########################


def blender_image_to_nparray(b3d_image):
    img_source_w = b3d_image.size[0]
    img_source_h = b3d_image.size[1]

    # Create an NP array filled with 0.5
    img_target = np.full((img_source_w, img_source_h, 3), .5)

    # Get all image pixels
    img_source_array = b3d_image.pixels[:]

    for height_i in range(0, img_source_h):

        for width_i in range(0, img_source_w):

            # Get pixel position in flat array
            colar = (width_i + (height_i * img_source_w)) * 4

            # Get color values at current "Pixel"
            r = round(img_source_array[colar - 4], 8)
            g = round(img_source_array[colar - 3], 8)
            b = round(img_source_array[colar - 2], 8)
            a = round(img_source_array[colar - 1], 8)

            # Fill NP Array with the collected values
            img_target[width_i][height_i][0] = r
            img_target[width_i][height_i][1] = g
            img_target[width_i][height_i][2] = b

    return(img_target)


def nparray_to_blender_image(image, image_out_x, image_out_y):

    # Create an image with the needed output dimensions
    blender_image = bpy.data.images.new(
        "MyImage", width=image_out_y, height=image_out_x)

    # Create Temporary pixel array to fill
    pixels = [None] * image_out_x * image_out_y

    for x in range(image_out_x):

        for y in range(image_out_y):

            # Get values from NP Array
            r = image[y][x][0]
            g = image[y][x][1]
            b = image[y][x][2]
            a = 1

            # Write values into temporary array
            pixels[(x * image_out_y) + y] = [r, g, b, a]

    # Flatten Pixel list into blender image object pixel list
    pixels = [chan for px in pixels for chan in px]

    # Set image data block pixel values to the flat temporary pixel array
    blender_image.pixels = pixels

    return


class WFC_OT_Runner(bpy.types.Operator):
    bl_idname = "object.wfc_ot_runner"
    bl_label = "Collapse"

    def execute(self, context):
        # Getting output dimensions from UI
        image_out_x = bpy.context.scene.wfc_vars.wfc_resultx
        image_out_y = bpy.context.scene.wfc_vars.wfc_resulty

        # Here 3D could be integrated
        image_out_z = 1

        # Set Grid size from the UI vars
        grid_size = (image_out_z, image_out_y, image_out_x)

        # Getting pattern dimensions from UI
        pat_x = bpy.context.scene.wfc_vars.wfc_patternx
        pat_y = bpy.context.scene.wfc_vars.wfc_patterny

        # Pattern preparation for 3D
        pat_z = 1

        # Set Pattern size from the UI vars
        pattern_size = (pat_z, pat_y, pat_x)

        # Get image name from UI and get Image Data Block
        img_name = bpy.context.scene.wfc_vars.wfc_images
        img = bpy.data.images[img_name]

        # Convert the Image Data Block to an NP Array fit for the collapse algorithm
        img_target = blender_image_to_nparray(img)

        # Expand image dimensions for use in WFC
        sample = load_sample(img_target)

        # Init WFC with the params
        wfc = WaveFunctionCollapse(grid_size, sample, pattern_size)

        # Running WFC, wfc.step could be used to generate animations
        wfc.run()

        # After running we request the image result
        image = wfc.get_image()

        # Take of a Dimension of the result
        if image.shape[0] == 1:
            image = np.squeeze(image, axis=0)

        # Write an image to blender from the NP Array
        nparray_to_blender_image(image, image_out_x, image_out_y)

        return {'FINISHED'}
