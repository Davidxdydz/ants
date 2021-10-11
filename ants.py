import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from utils import perf
from numba import jit

N = 10000
WIDTH = 1920
HEIGHT = 1080
SPEED = 1.5
EDGE_WIDTH = 10
ROTATION_FACTOR = 0.2
FADE_PER_STEP = 0.98
RADIUS = 5
FOV = np.pi/4
DIFFUSION = 0.2
SAMPLES = 10

domain = np.zeros((WIDTH, HEIGHT))

# domain[..., 270] = 1

# np.random.seed(0)

# [[x,y,direction]]
min_dir = min(HEIGHT, WIDTH)-EDGE_WIDTH
angles = np.random.uniform(0, np.pi*2, (N,))
lengths = np.sqrt(np.random.uniform(0, (min_dir/2)**2, (N,)))
xs = lengths * np.cos(angles) + WIDTH/2
ys = lengths * np.sin(angles) + HEIGHT / 2
ants = np.array([xs, ys, np.random.uniform(0, np.pi*2, (N,))]).T
# ants = np.array([np.random.random((N,))*(WIDTH-2*EDGE_WIDTH)+EDGE_WIDTH,
#                  np.random.random((N,))*(HEIGHT-2*EDGE_WIDTH)+EDGE_WIDTH,
#                  np.random.random((N,))*2*np.pi]).T
# ants = np.array([[500, 250, np.pi/3]], dtype=np.float64)

@perf()
@jit(nopython=True)
def wanted_direction_delta(domain, ants):
    angles = np.linspace(-FOV, FOV, SAMPLES)
    deltas = np.zeros((len(ants),), dtype=np.float64)
    for n, (x, y, a) in enumerate(ants):
        xs = (x + np.cos(angles+a) * RADIUS).astype(np.int64)
        ys = (y + np.sin(angles+a) * RADIUS).astype(np.int64)
        m = 0
        mi = 0
        for n,(x,y) in enumerate(zip(xs,ys)):
            if domain[x,y] > m:
                m = domain[x,y]
                mi = n
        deltas[n] = angles[mi]
    return deltas


@perf()
def step(ants):
    deltas = wanted_direction_delta(domain, ants)
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
def draw_positions(ants, domain):
    indices = ants[..., :2].T.astype(np.int64)
    domain[tuple(indices)] += 1


kernel = np.ones((3, 3))
kernel /= np.sum(kernel)

@perf()
def diffuse(domain):
    out = np.zeros_like(domain)
    d2 = average(out,domain)
    return domain*(1-DIFFUSION) + d2*DIFFUSION

@perf()
@jit(nopython= True)
def average(avgs,domain):
    for x in range(1,avgs.shape[0]-1):
        for y in range(1,avgs.shape[1]-1):
            avgs[x,y] = np.sum(domain[x-1:x+2,y-1:y+2])/9
    return avgs

@perf()
def fade(domain):
    domain = domain*FADE_PER_STEP
    return domain

@perf()
def getImage():
    global ants
    global domain
    step(ants)
    draw_positions(ants, domain)
    domain = fade(domain)
    domain = diffuse(domain)
    return domain.T


def animate(aspectRatio=1, cmap="afmhot", fps=30):
    fig = plt.figure(figsize=(int(aspectRatio*8), 8))
    plt.tight_layout()
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    dt = 1/fps * 1000  # in ms

    count = getImage()
    image = plt.imshow(count, cmap=cmap, animated=True,vmin = 0,vmax = 5)

    def update(zoom):
        count = getImage()
        image.set_array(count)
        return image,

    # TODO: tqdm on zooms is broken?? remove print from update
    return FuncAnimation(fig, update, range(1000), interval=dt, blit=True)


animation = animate()
plt.show()