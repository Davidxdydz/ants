import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib import image
from utils import perf, set_enabled
from numba import guvectorize, cuda
import time
import datetime

OUTPUT_PATH = "tmp.mp4"

# enable rendering and saving animation
SAVE = False


# number of ants
N = 1_000_000

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
RADIUS = 25

# ants look ahead from -FOV to FOV
FOV = np.pi/4

# how many points in front the ants sample
SAMPLES = 3

# the pheromones get averaged over adjacent pixels and the blended with the previous pheromone
# concentration, weighted by DIFFUSION
DIFFUSION = 0.5

ITERATIONS_PER_STEP = 10

# parameters for saving the animation
FRAMES = 5000
DPI = 100

# for time measuring
start = 0

# only output performance when not rendering
set_enabled(not SAVE)

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

# take one channel from image as obstacle
obstacles = image.imread("map.png")[..., 0].T

# smallest domain size, considering EDGEWIDTH
min_dir = min(HEIGHT, WIDTH)-2*EDGE_WIDTH
ants = spawn_ants_in_circle(N, min_dir/2, (WIDTH/2, HEIGHT/2))

# pheromone concentration
domain = np.zeros((WIDTH, HEIGHT), dtype=np.float32)

# precalculate the rotation matrices for the sample points in front of the ants
# saves some sin and cos in the heavy loops
look_angles = np.linspace(-FOV, FOV, SAMPLES)
look_rotations = np.array([[np.cos(a), -np.sin(a), np.sin(a), np.cos(a)]
                          for a in look_angles], dtype=np.float32)


@perf(name="step")
@guvectorize(['float32[:,:],float32[:,:],float32[:,:]'], "(n,m),(n,m),(k,l)", target='parallel')
def step(domain, view, ants):
    global look_rotations

    # for every ant
    for n, (x, y, vx, vy) in enumerate(ants):
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
            if view[x_int, y_int] > m:
                m = view[x_int, y_int]
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
        domain[cx, cy] += 1


@perf(name="diffuse")
@guvectorize(['float32[:,:],float32[:,:]'], '(n,m)->(n,m)', target="parallel")
def diffuse_and_evaporate(domain, avgs):
    # basically convolve with 3x3 ones
    for x in range(1, avgs.shape[0]-1):
        for y in range(1, avgs.shape[1]-1):
            avgs[x, y] = domain[x, y]

            avgs[x, y] += domain[x+1, y-1]
            avgs[x, y] += domain[x+1, y]
            avgs[x, y] += domain[x+1, y+1]

            avgs[x, y] += domain[x, y-1]
            avgs[x, y] += domain[x, y+1]

            avgs[x, y] += domain[x-1, y-1]
            avgs[x, y] += domain[x-1, y]
            avgs[x, y] += domain[x-1, y+1]

            # renormalize
            avgs[x, y] /= 9

            # blend the averaged domain with the original
            avgs[x, y] = (avgs[x, y]*DIFFUSION + domain[x, y]
                          * (1-DIFFUSION))*FADE_PER_STEP


@perf()
def get_image():
    global ants
    global domain
    global obstacles

    # mask the domain with obstacles
    view = domain * obstacles
    # perform one step for the ants
    step(domain, view, ants)
    # diffuse and evaporate ;)
    domain = diffuse_and_evaporate(domain)
    return domain.T


def animate(cmap="afmhot", fps=30):
    fig = plt.figure(figsize=(WIDTH/DPI, HEIGHT/DPI))
    plt.tight_layout()
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    dt = 1/fps * 1000  # in ms

    count = get_image()
    image = plt.imshow(count, cmap=cmap, animated=True, vmin=0, vmax=20)

    def update(n):
        if SAVE:
            # estimate remaining time when rendering
            global start
            t = time.time() - start
            per_frame = t/(n+1)
            remaining = per_frame*(FRAMES-n)
            r = datetime.timedelta(seconds=remaining)
            print(f"{n}/{FRAMES}: {r} remaining ({per_frame:.2f}s/per frame) ")
        for _ in range(ITERATIONS_PER_STEP):
            count = get_image()
        image.set_array(count)
        return image,

    return FuncAnimation(fig, update, None, interval=dt, blit=True, save_count=FRAMES)


animation = animate()

if SAVE:
    plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\david\\Documents\\ffmpeg-2021-10-11-git-90a0da9f14-essentials_build\\bin\\ffmpeg.exe'
    from matplotlib.animation import FFMpegWriter
    writermp4 = FFMpegWriter(fps=30, bitrate=80000)
    start = time.time()
    animation.save(OUTPUT_PATH, writer=writermp4, dpi=DPI)
else:
    plt.show()
