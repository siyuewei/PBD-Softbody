from mesh_data import simpleTet
from mesh_data import bunnyMesh
import taichi as ti
import taichi.math as tm
import numpy as np

ti.init()

data = bunnyMesh
#读取输入文件
numParticles = len(data['verts']) // 3
numTets = len(data['tetIds']) // 4
numEdges = len(data['tetEdgeIds']) // 2
numSurfaces = len(data['tetSurfaceTriIds']) // 3

np_particles = np.array(data['verts'], dtype=float)
np_tets = np.array(data['tetIds'], dtype=int)
np_edges = np.array(data['tetEdgeIds'], dtype=int)
np_surfaces = np.array(data['tetSurfaceTriIds'], dtype=int)

np_particles = np_particles.reshape(-1,3)
np_tets = np_tets.reshape(-1,4)
np_edges = np_edges.reshape(-1,2)
np_surfaces = np_surfaces.reshape(-1,3)

particles = ti.Vector.field(3, float, numParticles)
tets = ti.Vector.field(4, int, numTets)
edges = ti.Vector.field(2, int, numEdges)
surfaces = ti.Vector.field(3, int, numSurfaces)

particles.from_numpy(np_particles)
tets.from_numpy(np_tets)
edges.from_numpy(np_edges)
surfaces.from_numpy(np_surfaces)

# 计算初始状态下：
# 1.四面体的体积
# 2.边的长度
restVolumn = ti.field(float,numTets)
restLen = ti.field(float,numEdges)

@ti.func
def tetVolume(i):
    id = tm.ivec4(-1,-1,-1,-1)
    for j in ti.static(range(4)):
        id[j] = tets[i][j]
    temp = (particles[id[1]] - particles[id[0]]).cross(particles[id[2]] - particles[id[0]])
    res = temp.dot(particles[id[3]] - particles[id[0]])
    res *= 1.0/6.0
    return res

@ti.kernel
def init_physics():
    for i in tets:
        # point_0 = particles[tets[i][0]]
        # point_1 = particles[tets[i][1]]
        # point_2 = particles[tets[i][2]]
        # point_3 = particles[tets[i][3]]
        # volumn = (((point_1 - point_0).cross(point_2 - point_0)).dot(point_3 - point_0)) / 6.0
        restVolumn[i] = tetVolume(i)
    
    for i in edges:
        point_0 = particles[edges[i][0]]
        point_1 = particles[edges[i][1]]
        len = (point_0 - point_1).norm()
        restLen[i] = len

init_physics()

# 根据四面体体积设定每个点的质量倒数
invMass = ti.field(float, numParticles)

@ti.kernel
def init_invMass():
    for i in tets:
        volumn = restVolumn[i]
        tMass = 0.0
        if volumn > 0.0:
            tMass = 1.0 / volumn
        for j in ti.static(range(4)):
            invMass[tets[i][j]] += tMass

init_invMass()