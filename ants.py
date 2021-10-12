import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from utils import perf
from numba import jit, vectorize, guvectorize, float32
import time
import datetime
import math

N = 100000
WIDTH = 1920
HEIGHT = 1080
SPEED = 1.5
EDGE_WIDTH = 20
ROTATION_FACTOR = 0.5
FADE_PER_STEP = 0.98
RADIUS = 20
FOV = np.pi/4
DIFFUSION = 0.5
SAMPLES = 3
ITERATIONS_PER_STEP = 1
FRAMES = 5000
dpi = 100

np.random.seed(0)


# domain[..., 270] = 1

# np.random.seed(0)

# [[x,y,direction]]
min_dir = min(HEIGHT, WIDTH)-EDGE_WIDTH
angles = np.random.uniform(0, np.pi*2, (N,))
lengths = np.sqrt(np.random.uniform(0, (min_dir/2)**2, (N,)))
xs = lengths * np.cos(angles) + WIDTH/2
ys = lengths * np.sin(angles) + HEIGHT / 2
ants = np.array([xs, ys, np.random.uniform(
    0, np.pi*2, (N,))]).T.astype(np.float32)
# ants = np.array([np.random.random((N,))*(WIDTH-2*EDGE_WIDTH)+EDGE_WIDTH,
#                  np.random.random((N,))*(HEIGHT-2*EDGE_WIDTH)+EDGE_WIDTH,
#                  np.random.random((N,))*2*np.pi]).T
# ants = np.array([[500, 250, np.pi/3]], dtype=np.float64)

domain = np.zeros((WIDTH, HEIGHT), dtype=np.float32)
diffusion_buffer = np.zeros_like(domain, dtype=np.float32)
deltas_buffer = np.zeros((len(ants),), dtype=np.float32)

angles = np.linspace(-FOV, FOV, SAMPLES)


@vectorize()
def wanted_direction_delta(domain, x, y, a):
    global angles
    xs = (x + np.cos(angles+a) * RADIUS).astype(np.int32)
    ys = (y + np.sin(angles+a) * RADIUS).astype(np.int32)
    m = domain[xs[0], ys[0]]
    mi = 0
    for i, (x, y) in enumerate(zip(xs, ys)):
        if domain[x, y] > m:
            m = domain[x, y]
            mi = i
    return angles[mi]


@perf(name="dirs")
@guvectorize(['float32[:,:],float32[:,:],float32[:]'], "(n,m),(k,l) -> (k)", target='parallel')
def wanted_direction_deltas_vectorized(domain, ants, buffer):
    global angles
    for n, (x, y, a) in enumerate(ants):
        m = -1
        mi = 0
        for i, da in enumerate(angles):
            x_int = int(x + math.cos(a + da)*RADIUS)
            y_int = int(y + math.sin(a+da)*RADIUS)
            if domain[x_int, y_int] > m:
                m = domain[x_int, y_int]
                mi = i
        buffer[n] = angles[mi]


@perf()
def step_ants(ants,deltas):
    ants[..., 2] += deltas * np.random.random((len(ants,))) * ROTATION_FACTOR

    ants[..., 0] += np.cos(ants[..., 2])*SPEED
    ants[..., 1] += np.sin(ants[..., 2])*SPEED

    mask = ants[..., 0] < EDGE_WIDTH
    ants[..., 0][mask] = EDGE_WIDTH
    ants[..., 2][mask] = np.pi - ants[..., 2][mask]

    mask = ants[..., 0] >= WIDTH - EDGE_WIDTH
    ants[..., 0][mask] = WIDTH-1 - EDGE_WIDTH
    ants[..., 2][mask] = np.pi - ants[..., 2][mask]

    mask = ants[..., 1] < EDGE_WIDTH
    ants[..., 1][mask] = EDGE_WIDTH
    ants[..., 2][mask] = 2*np.pi - ants[..., 2][mask]

    mask = ants[..., 1] >= HEIGHT-EDGE_WIDTH
    ants[..., 1][mask] = HEIGHT-1 - EDGE_WIDTH
    ants[..., 2][mask] = 2*np.pi - ants[..., 2][mask]


@perf()
def step(ants):
    deltas = wanted_direction_deltas_vectorized(
        domain, ants)
    step_ants(ants,deltas)
    

@perf()
def draw_positions(ants, domain):
    indices = ants[..., :2].T.astype(np.int32)
    domain[tuple(indices)] += 1


kernel = np.ones((3, 3))
kernel /= np.sum(kernel)


@perf(name="diffuse")
@guvectorize(['float32[:,:],float32[:,:]'], '(n,m)->(n,m)', target="parallel")
def diffuse_and_evaporate(domain, avgs):
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

            avgs[x, y] /= 9
            avgs[x, y] = (avgs[x, y]*DIFFUSION + domain[x, y]
                          * (1-DIFFUSION))*FADE_PER_STEP


start = 0


@perf()
def getImage():
    global ants
    global domain
    step(ants)
    draw_positions(ants, domain)
    domain = diffuse_and_evaporate(domain)
    return domain.T


def animate(aspectRatio=1, cmap="afmhot", fps=30):
    fig = plt.figure(figsize=(WIDTH/dpi, HEIGHT/dpi))
    plt.tight_layout()
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    dt = 1/fps * 1000  # in ms

    count = getImage()
    image = plt.imshow(count, cmap=cmap, animated=True, vmin=0, vmax=8)

    def update(n):
        global start
        # t = time.time() - start
        # per_frame = t/(n+1)
        # remaining = per_frame*(FRAMES-n)
        # r = datetime.timedelta(seconds = remaining)
        # print(f"{n}/{FRAMES}: {r} remaining ({per_frame:.2f}s/per frame) ")
        for _ in range(ITERATIONS_PER_STEP):
            count = getImage()
        image.set_array(count)
        return image,

    return FuncAnimation(fig, update, None, interval=dt, blit=True, save_count=FRAMES)


animation = animate()
plt.show()
# plt.rcParams['animation.ffmpeg_path'] = 'C:\\Users\\david\\Documents\\ffmpeg-2021-10-11-git-90a0da9f14-essentials_build\\bin\\ffmpeg.exe'
# from matplotlib.animation import FFMpegWriter
# writermp4 = FFMpegWriter(fps=30,bitrate=80000)
# start = time.time()
# animation.save("test3.mp4",writer = writermp4,dpi= dpi)
