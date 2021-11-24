import time
import numpy as np
from utils import perf
from numba import cuda
import cv2

WINDOW_NAME = "pheromone concentration"

# number of ants
N = 2_000_000

# resolution
WIDTH = 1920
HEIGHT = 1080

# movement speed of the ants per step
SPEED = 1.5

# ants will be "reflected" EDGE_WIDTH  pixels before reaching the border of the domain.
# used so that the "sensing" of the ants does not cause an out of bounds are
EDGE_WIDTH = 25

# mulitplicative factor applied to the pheromone density after every step
FADE_PER_STEP = 0.97

# distance at which the ants sample the phermone density
RADIUS = 5

# ants look ahead from -FOV to FOV
FOV = np.pi/4

# how many points in front the ants sample
SAMPLES = 5

# the pheromones get averaged over adjacent pixels and the blended with the previous pheromone
# concentration, weighted by DIFFUSION
DIFFUSION = 0.8

ITERATIONS_PER_STEP = 1

# ensure reproducibility
np.random.seed(0)


def spawn_ants_in_circle(count, radius, center):
    # uniform random positions in a circle for ants
    position_angles = np.random.uniform(0, np.pi*2, (count,))
    # rescaling to ensure uniform distribution
    lengths = np.sqrt(np.random.uniform(0, radius**2, (count,)))
    xs = lengths * np.cos(position_angles) + center[0]
    ys = lengths * np.sin(position_angles) + center[1]

    # velocities for all ants
    direction_angles = np.random.uniform(0, np.pi*2, (count,))
    vxs = np.cos(direction_angles)
    vys = np.sin(direction_angles)
    ants = np.array([xs, ys, vxs, vys]).T
    return ants.astype(np.float32)


# smallest domain size, considering EDGEWIDTH
min_dir = min(HEIGHT, WIDTH)-2*EDGE_WIDTH
ants = spawn_ants_in_circle(N, min_dir/2, (WIDTH/2, HEIGHT/2))
ants = cuda.to_device(ants)

# pheromone concentration
domain = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

device_buffer_1 = cuda.to_device(domain)
device_buffer_2 = cuda.device_array_like(device_buffer_1)

pic_buffer = cuda.device_array((HEIGHT, WIDTH, 3), dtype=np.float32)

# precalculate the rotation matrices for the sample points in front of the ants
# saves some sin and cos in the heavy loops
look_angles = np.linspace(-FOV, FOV, SAMPLES)
look_rotations = np.array([[np.cos(a), -np.sin(a), np.sin(a), np.cos(a)]
                          for a in look_angles], dtype=np.float32)


@cuda.jit
def step_kernel(domain, view, ants):
    global look_rotations
    n = cuda.grid(1)

    if n >= len(ants):
        return

    if n >= len(ants):
        return
    (x, y, vx, vy) = ants[n]
    m = -1
    vxd = 0
    vyd = 0
    # find the highest pheromone concentration ahead
    for a, b, c, d in look_rotations:
        # rotation via precalculated rotation matrices
        vxn = vx*a+vy*c
        vyn = vx*b+vy*d
        x_int = int(x + vxn * RADIUS)
        y_int = int(y + vyn * RADIUS)
        if view[y_int, x_int] > m:
            m = view[y_int, x_int]
            vxd = vxn
            vyd = vyn

    # update the velocities and positions
    ants[n, 2] = vxd
    ants[n, 3] = vyd
    ants[n, 0] += ants[n, 2]*SPEED
    ants[n, 1] += ants[n, 3]*SPEED

    # bounce ants of domain bounds
    if ants[n, 0] < EDGE_WIDTH:
        ants[n, 0] = EDGE_WIDTH
        ants[n, 2] = - ants[n, 2]

    if ants[n, 0] >= WIDTH - EDGE_WIDTH:
        ants[n, 0] = WIDTH-1 - EDGE_WIDTH
        ants[n, 2] = - ants[n, 2]

    if ants[n, 1] < EDGE_WIDTH:
        ants[n, 1] = EDGE_WIDTH
        ants[n, 3] = - ants[n, 3]

    if ants[n, 1] >= HEIGHT-EDGE_WIDTH:
        ants[n, 1] = HEIGHT - 1 - EDGE_WIDTH
        ants[n, 3] = - ants[n, 3]

    # increase pheromone concentration at ants position
    cx, cy = int(ants[n, 0]), int(ants[n, 1])
    domain[cy, cx] += 1


# @perf(skip=1)
def step_cuda(domain, view, ants):
    per_thread = 1
    threadsperblock = 128
    blockspergrid = int(np.ceil(N/threadsperblock/per_thread))
    view = cuda.to_device(view)
    step_kernel[blockspergrid, threadsperblock](domain, view, ants)


@cuda.jit
def diffusion_kernel(domain, output,pic_out,render):
    x, y = cuda.grid(2)

    if x == 0 or y == 0 or x >= domain.shape[1]-1 or y >= domain.shape[0]-1:
        return

    result = domain[y, x]

    result += domain[y+1, x-1]
    result += domain[y+1, x]
    result += domain[y+1, x+1]

    result += domain[y, x-1]
    result += domain[y, x+1]

    result += domain[y-1, x-1]
    result += domain[y-1, x]
    result += domain[y-1, x+1]

    result /= 9
    result = (result*DIFFUSION + domain[y, x]
                    * (1-DIFFUSION))*FADE_PER_STEP
    output[y, x] = result
    if render:
        pic_out[y, x, 0] = result/512  # b
        pic_out[y, x, 1] = 0  # g
        pic_out[y, x, 2] = result/512  # r


# @perf(skip=1)
def diffuse_evaporate_render(domain,output, pic_out,render):
    threads_per_block_x = 16
    threads_per_block_y = 8
    blocks_per_grid_x = int(np.ceil(WIDTH / threads_per_block_x))
    blocks_per_grid_y = int(np.ceil(HEIGHT / threads_per_block_y))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    threads_per_block = (threads_per_block_x,threads_per_block_y)
    diffusion_kernel[blocks_per_grid, threads_per_block](domain, output, pic_out,render)


counter = 0


@perf(skip=1)
def get_image(steps):
    global ants
    global domain

    global device_buffer_1
    global device_buffer_2

    global counter

    for i in range(steps):
        # use two buffers to limit memory transfer
        if counter % 2 == 0:
            # perform one step for the ants
            device_buffer_1.copy_to_device(device_buffer_2)
            step_cuda(device_buffer_1, device_buffer_2, ants)
            # diffuse and evaporate ;)
            diffuse_evaporate_render(device_buffer_1,device_buffer_2, pic_buffer,i == steps-1)
            counter += 1
        else:
            # perform one step for the ants
            device_buffer_2.copy_to_device(device_buffer_1)
            step_cuda(device_buffer_2, device_buffer_1, ants)
            # diffuse and evaporate ;)
            diffuse_evaporate_render(device_buffer_2,device_buffer_1, pic_buffer,i == steps-1)
            counter += 1
    return pic_buffer.copy_to_host()


def animate(cmap="afmhot", target_fps=30):
    n = 0
    t = 0
    im = get_image(ITERATIONS_PER_STEP)
    start = time.time()
    while True:
        im = get_image(ITERATIONS_PER_STEP)
        cv2.imshow(WINDOW_NAME, im)
        n += 1
        t = time.time() - start
        print(f"{n/t:.2f} fps\n")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


animate()
