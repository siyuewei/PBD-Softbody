import taichi as ti
from ReadMesh import * 
import taichi.math as tm
import os
import shutil

fps = 60.0
substeps = 10
dt = ( 1.0 / fps / substeps)
edgeCompliance = 100.0
volumeCompliance = 0.0

gravity = ti.Vector([0.0, -9.8, 0.0])
ground_y = -0.5

surfaces_show = ti.field(int, 3*numSurfaces)
surfaces_show.from_numpy(np_surfaces.reshape(-1,))

# # 用于记录鼠标是否按下
# mouse_pressed = False
# # 用于记录鼠标按下时的坐标（二维屏幕坐标）
# mouse_down_pos = tm.vec2(0, 0)
# # 用于记录被拖拽的粒子索引，如果没有则为 -1
# dragged_particle_index = -1
# # 一个较小的距离阈值，用于判断鼠标点击是否命中粒子（可根据实际情况调整）
# click_distance_threshold = 800 

##### colors
colors = ti.Vector.field(3, float, numParticles)
@ti.kernel
def init_colors():
    for i in colors:
        color = ti.Vector([ti.random(),ti.random(),ti.random()])
        colors[i] = color
        # print(color)
init_colors()

##### velocity
velocity = ti.Vector.field(3, float, numParticles)
@ti.kernel
def init_velocity():
    for i in velocity:
        v = [0.0, 0.0, 0.0]
        velocity[i] = v
init_velocity()

preParticles = ti.Vector.field(3, float, numParticles)
@ti.kernel
def preSolve():
    for i in velocity:
        velocity[i] += gravity * dt
        preParticles[i] = particles[i]
        particles[i] += velocity[i] * dt
        
        if particles[i].y < ground_y:
            particles[i] = preParticles[i]
            particles[i].y = ground_y

@ti.kernel
def solveEdge():
    alpha = edgeCompliance / dt / dt
    for i in edges:
        point_0 = particles[edges[i][0]]
        point_1 = particles[edges[i][1]]
        w_0 = invMass[edges[i][0]]
        w_1 = invMass[edges[i][1]]
        c_0 = (point_0 - point_1) / (point_0 - point_1).norm()
        c_1 = (point_1 - point_0) / (point_0 - point_1).norm()
        lam = -((point_0 - point_1).norm() - restLen[i]) / (alpha + w_0 * ((c_0.norm()) ** 2) + w_1 * ((c_1.norm()) ** 2))

        particles[edges[i][0]] += lam * w_0 * c_0
        particles[edges[i][1]] += lam * w_1 * c_1

#这段代码的意思和下面的同名函数是一样的，但是性能太差导致跑不出结果
# @ti.kernel
# def solveVolume():
#     for i in tets:
#         point_0 = particles[tets[i][0]]
#         point_1 = particles[tets[i][1]]
#         point_2 = particles[tets[i][2]]
#         point_3 = particles[tets[i][3]]
#         w_0 = invMass[edges[i][0]]
#         w_1 = invMass[edges[i][1]]
#         w_2 = invMass[edges[i][2]]
#         w_3 = invMass[edges[i][3]]
#         volumn = (((point_1 - point_0).cross(point_2 - point_0)).dot(point_3 - point_0)) / 6.0
#         c_0 = (point_3 - point_1).cross(point_2 - point_1)    
#         c_1 = (point_2 - point_0).cross(point_3 - point_0)    
#         c_2 = (point_3 - point_0).cross(point_1 - point_0)    
#         c_3 = (point_1 - point_0).cross(point_2 - point_0)
#         lam = -6 * (volumn - restVolumn[i]) / (w_0 * ((c_0.norm()) ** 2) + w_1 * ((c_1.norm()) ** 2) + w_2 * ((c_2.norm()) ** 2) + w_3 * ((c_3.norm()) ** 2))
        
#         particles[tets[i][0]] += lam * w_0 * c_0
#         particles[tets[i][1]] += lam * w_1 * c_1
#         particles[tets[i][2]] += lam * w_2 * c_2
#         particles[tets[i][3]] += lam * w_3 * c_3

@ti.kernel
def solveVolume():
    alpha = volumeCompliance / dt / dt
    grads = [tm.vec3(0,0,0), tm.vec3(0,0,0), tm.vec3(0,0,0), tm.vec3(0,0,0)]
    
    for i in range(numTets):
        id = tm.ivec4(-1,-1,-1,-1)
        for j in ti.static(range(4)):
            id[j] = tets[i][j]
        grads[0] = (particles[id[3]] - particles[id[1]]).cross(particles[id[2]] - particles[id[1]])
        grads[1] = (particles[id[2]] - particles[id[0]]).cross(particles[id[3]] - particles[id[0]])
        grads[2] = (particles[id[3]] - particles[id[0]]).cross(particles[id[1]] - particles[id[0]])
        grads[3] = (particles[id[1]] - particles[id[0]]).cross(particles[id[2]] - particles[id[0]])

        w = 0.0
        for j in ti.static(range(4)):
            w += invMass[id[j]] * (grads[j].norm())**2

        vol = (((particles[tets[i][1]] - particles[tets[i][0]]).cross(particles[tets[i][2]] - particles[tets[i][0]])).dot(particles[tets[i][3]] - particles[tets[i][0]])) / 6.0
        C = (vol - restVolumn[i]) * 6.0
        s = -C /(w + alpha)
        
        for j in ti.static(range(4)):
            particles[tets[i][j]] += grads[j] * s * invMass[id[j]]


def solve():
    solveEdge()
    solveVolume()

@ti.kernel
def postSolve():
    for i in velocity:
        velocity[i] = (particles[i] - preParticles[i]) / dt

def substep():
    preSolve()
    solve()
    postSolve()

window = ti.ui.Window("Taichi PBD Softbody on GGUI",(1024,1024),vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1,1,1))
scene = window.get_scene()
camera = ti.ui.Camera()
camera_pos = ti.Vector([3.0, 0.0, 0.0])
camera_lookat = ti.Vector([0.0, 0.0, 0.0])

is_paused = False

is_record = False
if is_record:
    frames = []
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(parent_dir,'results')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    else:
        shutil.rmtree(result_dir)
        os.makedirs(result_dir)
    video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=fps, automatic_build=False)

all_t = 3.0
current_t = 0.0
while window.running:
    camera.position(camera_pos.x, camera_pos.y,camera_pos.z)
    camera.lookat(camera_lookat.x, camera_lookat.y, camera_lookat.z)
    scene.set_camera(camera)

    for e in window.get_events(ti.ui.PRESS):  # 检测窗口事件
        if e.key == ti.ui.SPACE:  # 如果按下空格键
            is_paused = not is_paused  # 切换暂停状态

    if not is_paused:
        for i in range(substeps):
            substep()
            current_t += dt

    if current_t > all_t: 
        break

    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(particles,
               indices=surfaces_show,
               per_vertex_color=colors,
               two_sided=True)

    canvas.scene(scene)
    window.show()

    if is_record:
        img_np = window.get_image_buffer_as_numpy()
        frames.append(img_np)

        video_manager.write_frame(img_np)
        print(f'\rFrame {len(frames)}/{all_t * fps } is recorded', end='') 

if is_record:
    print()
    print('Exporting.mp4 and.gif videos...')
    video_manager.make_video(gif=True, mp4=True)
    print(f'MP4 video is saved to {video_manager.get_output_filename(".mp4")}')
    print(f'GIF video is saved to {video_manager.get_output_filename(".gif")}')