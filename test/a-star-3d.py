## A star Algorithm

import heapq
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Node:
    def __init__(self, position, cost=0, parent=None):
        self.position = position
        self.cost = cost
        self.parent = parent

    def __lt__(self, other):
        return self.cost < other.cost

def heuristic(current, goal):
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1]) + abs(current[2] - goal[2])

def is_valid_position(position, grid_shape):
    return 0 <= position[0] < grid_shape[0] and 0 <= position[1] < grid_shape[1] and 0 <= position[2] < grid_shape[2]

def get_neighbors(current, grid_shape):
    x, y, z = current
    neighbors = []

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue

                neighbor = (x + dx, y + dy, z + dz)
                if is_valid_position(neighbor, grid_shape):
                    neighbors.append(neighbor)

    return neighbors

def visualize_path(voxel_grid, path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = np.where(voxel_grid == 1)
    ax.scatter(x, y, z, c='black', marker='s', label='Obstacle')

    x, y, z = zip(*path)
    ax.plot(x, y, z, marker='o', linestyle='-', color='r', label='Path')

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    plt.legend()
    plt.show()

def astar():
    #3D voxel
    voxel_grid = np.zeros((10, 10, 10))
    voxel_grid[2:8, 2:8, 2:8] = 1  # Obstacle in voxel

    start = (0, 0, 0)
    goal = (9, 9, 9)
    heap = [Node(start)]
    visited = set()

    while heap:
        current_node = heapq.heappop(heap)
        current_position = current_node.position

        if current_position == goal:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent

            path.reverse()
            return path

        if current_position in visited:
            continue

        visited.add(current_position)

        for neighbor_position in get_neighbors(current_position, voxel_grid.shape):
            if voxel_grid[neighbor_position] == 1:
                continue

            neighbor_node = Node(neighbor_position)
            #f(n)=g(n)+h(n)
            neighbor_node.cost = current_node.cost + heuristic(neighbor_position, goal)
            neighbor_node.parent = current_node

            heapq.heappush(heap, neighbor_node)

    return None

path = astar()

print("Simplest Path:", path)
visualize_path(voxel_grid, path)
