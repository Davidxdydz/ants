import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from utils import perf
from numba import jit, vectorize, guvectorize, float32
import time
import datetime
import math

N = 1000000
WIDTH = 1920
HEIGHT = 1080
SPEED = 1.5
EDGE_WIDTH = 25
FADE_PER_STEP = 0.97
RADIUS = 25
FOV = np.pi/4
DIFFUSION = 0.5
SAMPLES = 3
ITERATIONS_PER_STEP = 10
FRAMES = 5000
dpi = 100

np.random.seed(0)

# [[x,y,direction]]
min_dir = min(HEIGHT, WIDTH)-2*EDGE_WIDTH
position_angles = np.random.uniform(0, np.pi*2, (N,))
lengths = np.sqrt(np.random.uniform(0, (min_dir/2)**2, (N,)))
xs = lengths * np.cos(position_angles) + WIDTH/2
ys = lengths * np.sin(position_angles) + HEIGHT / 2
direction_angles = np.random.uniform(0, np.pi*2, (N,))
vxs = np.cos(direction_angles)
vys = np.sin(direction_angles)
ants = np.array([xs, ys, vxs, vys]).T.astype(np.float32)
# ants = np.array([np.random.random((N,))*(WIDTH-2*EDGE_WIDTH)+EDGE_WIDTH,
#                  np.random.random((N,))*(HEIGHT-2*EDGE_WIDTH)+EDGE_WIDTH,
#                  np.random.random((N,))*2*np.pi]).T
# ants = np.array([[500, 250, np.pi/3]], dtype=np.float64)

domain = np.zeros((WIDTH, HEIGHT), dtype=np.float32)

look_angles = np.linspace(-FOV, FOV, SAMPLES)
look_rotations = np.array([[np.cos(a), -np.sin(a), np.sin(a), np.cos(a)]
                          for a in look_angles], dtype=np.float32)


@perf(name="step")
@guvectorize(['float32[:,:],float32[:,:]'], "(n,m),(k,l)", target='parallel')
def step(domain, ants):
    global look_rotations
    for n, (x, y, vx, vy) in enumerate(ants):
        m = -1
        vxd = 0
        vyd = 0
        for a, b, c, d in look_rotations:
            vxn = vx*a+vy*c
            vyn = vx*b+vy*d
            x_int = int(x + vxn * RADIUS)
            y_int = int(y + vyn * RADIUS)
            if domain[x_int, y_int] > m:
                m = domain[x_int, y_int]
                vxd = vxn
                vyd = vyn
        ants[n, 2] = vxd
        ants[n, 3] = vyd
        ants[n, 0] += ants[n, 2]*SPEED
        ants[n, 1] += ants[n, 3]*SPEED

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

        cx, cy = int(ants[n, 0]), int(ants[n, 1])
        domain[cx, cy] += 1


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
    step(domain, ants)
    #draw_positions(ants, domain)
    domain = diffuse_and_evaporate(domain)
    # diffuse_and_evaporate.parallel_diagnostics(level=4)
    return domain.T


def animate(cmap="afmhot", fps=30):
    fig = plt.figure(figsize=(WIDTH/dpi, HEIGHT/dpi))
    plt.tight_layout()
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    dt = 1/fps * 1000  # in ms

    count = getImage()
    image = plt.imshow(count, cmap=cmap, animated=True, vmin=0, vmax=20)

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
