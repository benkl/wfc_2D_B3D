import bpy

import random
import math
import sys

UP = (0, 1)
LEFT = (-1, 0)
DOWN = (0, -1)
RIGHT = (1, 0)
DIRS = [UP, DOWN, LEFT, RIGHT]


class CompatibilityOracle(object):

    """The CompatibilityOracle class is responsible for telling us
    which combinations of tiles and directions are compatible. It's
    so simple that it perhaps doesn't need to be a class, but I think
    it helps keep things clear.
    """

    def __init__(self, data):
        self.data = data

    def check(self, tile1, tile2, direction):
        return (tile1, tile2, direction) in self.data


class Wavefunction(object):

    """The Wavefunction class is responsible for storing which tiles
    are permitted and forbidden in each location of an output image.
    """

    @staticmethod
    def mk(size, weights):
        """Initialize a new Wavefunction for a grid of `size`,
        where the different tiles have overall weights `weights`.
        Arguments:
        size -- a 2-tuple of (width, height)
        weights -- a dict of tile -> weight of tile
        """
        coefficients = Wavefunction.init_coefficients(size, weights.keys())
        return Wavefunction(coefficients, weights)

    @staticmethod
    def init_coefficients(size, tiles):
        """Initializes a 2-D wavefunction matrix of coefficients.
        The matrix has size `size`, and each element of the matrix
        starts with all tiles as possible. No tile is forbidden yet.
        NOTE: coefficients is a slight misnomer, since they are a
        set of possible tiles instead of a tile -> number/bool dict. This
        makes the code a little simpler. We keep the name `coefficients`
        for consistency with other descriptions of Wavefunction Collapse.
        Arguments:
        size -- a 2-tuple of (width, height)
        tiles -- a set of all the possible tiles
        Returns:
        A 2-D matrix in which each element is a set
        """
        coefficients = []

        for x in range(size[0]):
            row = []
            for y in range(size[1]):
                row.append(set(tiles))
            coefficients.append(row)

        return coefficients

    def __init__(self, coefficients, weights):
        self.coefficients = coefficients
        self.weights = weights

    def get(self, co_ords):
        """Returns the set of possible tiles at `co_ords`"""
        x, y = co_ords
        return self.coefficients[x][y]

    def get_collapsed(self, co_ords):
        """Returns the only remaining possible tile at `co_ords`.
        If there is not exactly 1 remaining possible tile then
        this method raises an exception.
        """
        opts = self.get(co_ords)
        assert(len(opts) == 1)
        return next(iter(opts))

    def get_all_collapsed(self):
        """Returns a 2-D matrix of the only remaining possible
        tiles at each location in the wavefunction. If any location
        does not have exactly 1 remaining possible tile then
        this method raises an exception.
        """
        width = len(self.coefficients)
        height = len(self.coefficients[0])

        collapsed = []
        for x in range(width):
            row = []
            for y in range(height):
                row.append(self.get_collapsed((x, y)))
            collapsed.append(row)

        return collapsed

    def shannon_entropy(self, co_ords):
        """Calculates the Shannon Entropy of the wavefunction at
        `co_ords`.
        """
        x, y = co_ords

        sum_of_weights = 0
        sum_of_weight_log_weights = 0
        for opt in self.coefficients[x][y]:
            weight = self.weights[opt]
            sum_of_weights += weight
            sum_of_weight_log_weights += weight * math.log(weight)

        return math.log(sum_of_weights) - (sum_of_weight_log_weights / sum_of_weights)

    def is_fully_collapsed(self):
        """Returns true if every element in Wavefunction is fully
        collapsed, and false otherwise.
        """
        for x, row in enumerate(self.coefficients):
            for y, sq in enumerate(row):
                if len(sq) > 1:
                    return False

        return True

    def collapse(self, co_ords):
        """Collapses the wavefunction at `co_ords` to a single, definite
        tile. The tile is chosen randomly from the remaining possible tiles
        at `co_ords`, weighted according to the Wavefunction's global
        `weights`.
        This method mutates the Wavefunction, and does not return anything.
        """
        x, y = co_ords
        opts = self.coefficients[x][y]
        valid_weights = {tile: weight for tile,
                         weight in self.weights.items() if tile in opts}

        total_weights = sum(valid_weights.values())
        rnd = random.random() * total_weights

        chosen = None
        for tile, weight in valid_weights.items():
            rnd -= weight
            if rnd < 0:
                chosen = tile
                break

        self.coefficients[x][y] = set(chosen)

    def constrain(self, co_ords, forbidden_tile):
        """Removes `forbidden_tile` from the list of possible tiles
        at `co_ords`.
        This method mutates the Wavefunction, and does not return anything.
        """
        x, y = co_ords
        self.coefficients[x][y].remove(forbidden_tile)


class Model(object):

    """The Model class is responsible for orchestrating the
    Wavefunction Collapse algorithm.
    """

    def __init__(self, output_size, weights, compatibility_oracle):
        self.output_size = output_size
        self.compatibility_oracle = compatibility_oracle

        self.wavefunction = Wavefunction.mk(output_size, weights)

    def run(self):
        """Collapses the Wavefunction until it is fully collapsed,
        then returns a 2-D matrix of the final, collapsed state.
        """
        while not self.wavefunction.is_fully_collapsed():
            self.iterate()

        return self.wavefunction.get_all_collapsed()

    def iterate(self):
        """Performs a single iteration of the Wavefunction Collapse
        Algorithm.
        """
        # 1. Find the co-ordinates of minimum entropy
        co_ords = self.min_entropy_co_ords()
        # 2. Collapse the wavefunction at these co-ordinates
        self.wavefunction.collapse(co_ords)
        # 3. Propagate the consequences of this collapse
        self.propagate(co_ords)

    def propagate(self, co_ords):
        """Propagates the consequences of the wavefunction at `co_ords`
        collapsing. If the wavefunction at (x,y) collapses to a fixed tile,
        then some tiles may not longer be theoretically possible at
        surrounding locations.
        This method keeps propagating the consequences of the consequences,
        and so on until no consequences remain.
        """
        stack = [co_ords]

        while len(stack) > 0:
            cur_coords = stack.pop()
            # Get the set of all possible tiles at the current location
            cur_possible_tiles = self.wavefunction.get(cur_coords)

            # Iterate through each location immediately adjacent to the
            # current location.
            for d in valid_dirs(cur_coords, self.output_size):
                other_coords = (cur_coords[0] + d[0], cur_coords[1] + d[1])

                # Iterate through each possible tile in the adjacent location's
                # wavefunction.
                for other_tile in set(self.wavefunction.get(other_coords)):
                    # Check whether the tile is compatible with any tile in
                    # the current location's wavefunction.
                    other_tile_is_possible = any([
                        self.compatibility_oracle.check(cur_tile, other_tile, d) for cur_tile in cur_possible_tiles
                    ])
                    # If the tile is not compatible with any of the tiles in
                    # the current location's wavefunction then it is impossible
                    # for it to ever get chosen. We therefore remove it from
                    # the other location's wavefunction.
                    if not other_tile_is_possible:
                        self.wavefunction.constrain(other_coords, other_tile)
                        stack.append(other_coords)

    def min_entropy_co_ords(self):
        """Returns the co-ords of the location whose wavefunction has
        the lowest entropy.
        """
        min_entropy = None
        min_entropy_coords = None

        width, height = self.output_size
        for x in range(width):
            for y in range(height):
                if len(self.wavefunction.get((x, y))) == 1:
                    continue

                entropy = self.wavefunction.shannon_entropy((x, y))
                # Add some noise to mix things up a little
                entropy_plus_noise = entropy - (random.random() / 1000)
                if min_entropy is None or entropy_plus_noise < min_entropy:
                    min_entropy = entropy_plus_noise
                    min_entropy_coords = (x, y)

        return min_entropy_coords


# def render_colors(matrix, colors):
#     """Render the fully collapsed `matrix` using the given `colors.
#     Arguments:
#     matrix -- 2-D matrix of tiles
#     colors -- dict of tile -> `colorama` color
#     """
#     for row in matrix:
#         output_row = []
#         for val in row:
#             color = colors[val]
#             output_row.append(color + val + colorama.Style.RESET_ALL)

#         print("".join(output_row))


def valid_dirs(cur_co_ord, matrix_size):
    """Returns the valid directions from `cur_co_ord` in a matrix
    of `matrix_size`. Ensures that we don't try to take step to the
    left when we are already on the left edge of the matrix.
    """
    x, y = cur_co_ord
    width, height = matrix_size
    dirs = []

    if x > 0:
        dirs.append(LEFT)
    if x < width-1:
        dirs.append(RIGHT)
    if y > 0:
        dirs.append(DOWN)
    if y < height-1:
        dirs.append(UP)

    return dirs


def parse_example_matrix(matrix):
    """Parses an example `matrix`. Extracts:

    1. Tile compatibilities - which pairs of tiles can be placed next
        to each other and in which directions
    2. Tile weights - how common different tiles are
    Arguments:
    matrix -- a 2-D matrix of tiles
    Returns:
    A tuple of:
    * A set of compatibile tile combinations, where each combination is of
        the form (tile1, tile2, direction)
    * A dict of weights of the form tile -> weight
    """
    compatibilities = set()
    matrix_width = len(matrix)
    matrix_height = len(matrix[0])

    weights = {}

    for x, row in enumerate(matrix):
        for y, cur_tile in enumerate(row):
            if cur_tile not in weights:
                weights[cur_tile] = 0
            weights[cur_tile] += 1

            for d in valid_dirs((x, y), (matrix_width, matrix_height)):
                other_tile = matrix[x+d[0]][y+d[1]]
                compatibilities.add((cur_tile, other_tile, d))

    return compatibilities, weights


def blender_transitions(wfc_result_array):
    output_tiles = []
    for x in range(0, len(wfc_result_array)):
        l_len = len(wfc_result_array[0])
        for y in range(0, l_len):
            blender_tile = ['L', 'L', 'L', 'L']
            if x - 1 >= 0:
                blender_tile[0] = wfc_result_array[x-1][y]
            if y + 1 < l_len:
                blender_tile[1] = wfc_result_array[x][y+1]
            if x + 1 < len(wfc_result_array):
                blender_tile[2] = wfc_result_array[x+1][y]
            if y - 1 >= 0:
                blender_tile[3] = wfc_result_array[x][y-1]
            output_tiles.append(blender_tile)

    return output_tiles


input_matrix = [
    ['L', 'L', 'L', 'L', 'L', 'L', 'L'],
    ['L', 'L', 'L', 'L', 'C', 'C', 'C'],
    ['L', 'L', 'L', 'L', 'C', 'G', 'C'],
    ['L', 'C', 'C', 'C', 'C', 'G', 'C'],
    ['L', 'C', 'G', 'C', 'C', 'G', 'C'],
    ['L', 'C', 'G', 'G', 'G', 'G', 'C'],
    ['L', 'C', 'G', 'G', 'G', 'G', 'C'],
    ['L', 'C', 'C', 'C', 'C', 'C', 'C'],
    ['L', 'L', 'L', 'L', 'L', 'L', 'L'],
]
input_matrix2 = [
    ['A', 'A', 'A', 'A'],
    ['A', 'A', 'A', 'A'],
    ['A', 'A', 'A', 'A'],
    ['A', 'C', 'C', 'A'],
    ['C', 'B', 'B', 'C'],
    ['C', 'B', 'B', 'C'],
    ['A', 'C', 'C', 'A'],
]

input_matrix3 = [
    ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'],
    ['A', 'B', 'B', 'D', 'D', 'D', 'D', 'A', 'G', 'G', 'G', 'A'],
    ['A', 'B', 'B', 'D', 'D', 'D', 'D', 'A', 'G', 'G', 'G', 'A'],
    ['A', 'A', 'A', 'A', 'A', 'D', 'D', 'A', 'G', 'G', 'G', 'A'],
    ['A', 'C', 'C', 'C', 'A', 'D', 'D', 'A', 'G', 'G', 'G', 'A'],
    ['A', 'C', 'C', 'C', 'A', 'D', 'D', 'A', 'G', 'G', 'G', 'A'],
    ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'],
    ['A', 'E', 'E', 'E', 'E', 'E', 'E', 'A', 'F', 'F', 'F', 'A'],
    ['A', 'E', 'E', 'E', 'E', 'E', 'E', 'A', 'F', 'F', 'F', 'A'],
    ['A', 'E', 'E', 'E', 'E', 'E', 'E', 'A', 'F', 'F', 'F', 'A'],
    ['A', 'E', 'E', 'E', 'E', 'E', 'E', 'A', 'F', 'F', 'F', 'A'],
    ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'],
]
input_matrix4 = [
    ['A', 'A', 'A', 'A', 'D', 'D', 'D', 'D'],
    ['A', 'B', 'B', 'A', 'D', 'E', 'E', 'D'],
    ['A', 'B', 'B', 'A', 'D', 'E', 'E', 'D'],
    ['A', 'A', 'A', 'A', 'D', 'E', 'E', 'D'],
    ['C', 'C', 'C', 'C', 'D', 'D', 'D', 'D'],
    ['C', 'F', 'F', 'C', 'A', 'A', 'A', 'A'],
    ['C', 'F', 'F', 'C', 'A', 'B', 'B', 'A'],
    ['C', 'F', 'F', 'C', 'A', 'A', 'A', 'A'],
    ['C', 'C', 'C', 'C', 'G', 'G', 'G', 'G'],
    ['D', 'D', 'D', 'D', 'G', 'H', 'H', 'G'],
    ['D', 'E', 'E', 'D', 'G', 'H', 'H', 'G'],
    ['D', 'D', 'D', 'D', 'G', 'G', 'G', 'G'],
]

input_matrix5 = [
    ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
    ['B', 'B', 'B', 'B', 'B', 'B', 'B', 'B'],
    ['A', 'B', 'B', 'B', 'B', 'B', 'A', 'B'],
    ['A', 'A', 'B', 'B', 'B', 'A', 'A', 'A'],
    ['A', 'A', 'A', 'B', 'A', 'A', 'A', 'A'],
    ['A', 'D', 'A', 'A', 'A', 'A', 'A', 'A'],
    ['D', 'D', 'D', 'A', 'A', 'A', 'A', 'A'],
    ['A', 'D', 'A', 'A', 'A', 'A', 'A', 'A'],
    ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'],
    ['A', 'A', 'A', 'A', 'A', 'C', 'C', 'A'],
    ['C', 'A', 'C', 'C', 'C', 'C', 'C', 'C'],
    ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C'],
]


# render_colors(output, colors)


class WFC_OT_Runner(bpy.types.Operator):
    bl_idname = "object.wfc_ot_runner"
    bl_label = "Modular WFC"

    def execute(self, context):

        compatibilities, weights = parse_example_matrix(input_matrix5)
        compatibility_oracle = CompatibilityOracle(compatibilities)
        model = Model((10, 50), weights, compatibility_oracle)
        output = model.run()
        transition_output = blender_transitions(output)
        print(output)
        print(get_tiles(output))
        block_placer(output)
        return {'FINISHED'}

# compatibilities, weights = parse_example_matrix(input_matrix)
# compatibility_oracle = CompatibilityOracle(compatibilities)
# model = Model((10, 50), weights, compatibility_oracle)
# output = model.run()
# transition_output = blender_transitions(output)
# print(output)
# print(len(transition_output), transition_output)


def get_tiles(tiles):
    temp_tiles = []
    l_len = len(tiles[0])
    for x in range(0, len(tiles)):
        for y in range(0, l_len):
            if tiles[x][y] not in temp_tiles:
                temp_tiles.append(tiles[x][y])
            else:
                pass
    return temp_tiles


def block_placer(output_array):
    # Find Collection
    seed = random.randrange(sys.maxsize)
    result_name = 'Result ' + str(seed)

    if (bpy.data.collections.find('Tiles') == True):
        tiles_collection = bpy.data.collections['Tiles']
    else:
        tiles_collection = bpy.data.collections.new('Tiles')

    bpy.context.scene.collection.children.link(tiles_collection)
    l_len = len(output_array[0])
    variants = get_tiles(output_array)
    result_collection = bpy.data.collections.new(result_name)
    bpy.context.scene.collection.children.link(result_collection)
    for t in range(0, len(variants)):
        bpy.ops.mesh.primitive_plane_add(location=(t, -5, 0), size=1)
        basemesh = bpy.context.object
        basemesh.name = variants[t]
        # bpy.data.collections['Tiles'].objects.link(basemesh)
        tiles_collection.objects.link(basemesh)
        # bpy.context.scene.collection.children.unlink(basemesh)
        print(t)

    for x in range(0, len(output_array)):
            # in the cell
        for y in range(0, l_len):
            srcobj = tiles_collection.objects[output_array[x][y]]
            if srcobj.type == 'MESH':
                print('mesh')
                copymesh = srcobj.copy()
                copymesh.name = output_array[x][y] + str((x + 1) * (y + 1))
                copymesh.data = srcobj.data
                # copymesh = bpy.context.object
                copymesh.location = (x, y, 0)
                result_collection.objects.link(copymesh)
            else:
                pass

        # set mesh name
    return
