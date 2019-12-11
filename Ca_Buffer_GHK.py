import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import math
import sys
import numpy
import random
import time
import steps.model as smodel
import steps.solver as solvmod
import steps.mpi.solver as mpisolvmod
import steps.geom as stetmesh
import steps.rng as srng
import steps.utilities.meshio as smeshio
import steps.utilities.meshctrl as meshctrl
import time

# from termcolor import colored
from utils.printProgressBar import printProgressBar


# The diffusion constant for our diffusing species (m^2/s)

# meshfile = 'meshes/cyl4_5K.inp'
# meshfile = 'meshes/cyl4_70Ktets.inp'
# meshfile = 'meshes/cyl4_20Ktets.inp'
meshfile = 'meshes/cyl4_10Ktets.inp'
DCST = 0.2e-10
cbaryc = [0, 0, 0]
SCALING_FACTOR = 1e-6 # whole to micro
AZ_RADIUS = 0.4 * SCALING_FACTOR
azCaChN = 150
print('SCALING_FACTOR: {}'.format(SCALING_FACTOR))


mesh, nodeproxy, tetproxy, triproxy = smeshio.importAbaqus(meshfile, SCALING_FACTOR)

tetgroups = tetproxy.blocksToGroups()
CS_tet_IDs = tetgroups['cytosol']
ER_tet_IDs = tetgroups['ER']
print("=======================================================")
print("                   GEOMETRY STATS:                     ")
print("=======================================================")
print("_______________________________________________________")

print('number of tets in CS: ', CS_tet_IDs.__len__())
print('number of tets in ER: ', ER_tet_IDs.__len__())
print('overlapping tets in CS and ER:', set(CS_tet_IDs).intersection(ER_tet_IDs))  

ERvol = sum([mesh.getTetVol(i) for i in ER_tet_IDs])
CSvol = sum([mesh.getTetVol(i) for i in CS_tet_IDs])

print('ER volume: {:.4f} um3'.format(ERvol * 1e18))
print('CS volume: {:.4f} um3'.format(CSvol * 1e18))
print('Total vol: {:.4f} um3'.format((CSvol + ERvol) * 1e18))

print("_______________________________________________________")

# get tets and tris in AZ:
triBARYC = []
surftris = mesh.getSurfTris()

surftrirads, in_az = [], []
for surftriID in surftris:
    baryc = mesh.getTriBarycenter(surftriID)
    triBARYC.append(baryc)
    r = math.sqrt(math.pow((baryc[0]-cbaryc[0]),2) \
                    + math.pow((baryc[1]-cbaryc[1]),2) \
                        + math.pow((baryc[2]-cbaryc[2]),2))
    surftrirads.append(r)
    in_az.append(True if r <= AZ_RADIUS else False)
zipped0 = zip(surftrirads, surftris, in_az)
zipped0 = sorted(zipped0, key=lambda x: x[0])
az_tris = [i[1] for i in zipped0 if i[0]<AZ_RADIUS and i[2]==True]

# get surface tetrahedrons within the active zone:
az_tets = [mesh.getTriTetNeighb(i)[0] for i in az_tris]

# get the triangles of those AZ tetrahedrons:
az_tet_tris = []
for az_tet in az_tets:
    az_quart = mesh.getTetTriNeighb(az_tet)
    for i in az_quart:
        az_tet_tris.append(i)
az_area = sum([mesh.getTriArea(i) for i in az_tris])
print('Tris in AZ: ', len(az_tris))
print('Tets in AZ: ', len(az_tets))
print('AZ area {:.4f} um2'.format(az_area * 1e12))
print("_______________________________________________________")

memb_area = sum([mesh.getTriArea(i) for i in surftris])
print('Memb. area {:.4f} um2'.format(memb_area * 1e12))
print("_______________________________________________________")

print('Furthest distances along axes (x, y, z)')
for ax in range(3):
    print('{:.4f} um'. format(max([triBARYC[i][ax] for i in range(len(triBARYC))]) * 1e6))
    
print('Origin (x, y, z)')
print(cbaryc)

print("_______________________________________________________")

CaSensTris = [zipped0[i][1] for i in range(6)]
NotCaSensTris = list(set(az_tris)-set(CaSensTris))
NotCaSensTris_area = sum([mesh.getTriArea(i) for i in NotCaSensTris])
print('NotCaSensTris:  {}'.format(len(NotCaSensTris)))
print('NotCaSensTris_area:  {:.4f} um2'.format(NotCaSensTris_area * 1e12))
print('Ca sensor tris: {}'.format(CaSensTris))
CaSensTets = [mesh.getTriTetNeighb(i)[0] for i in CaSensTris]
NotCaSensTets = list(set(az_tets)-set(CaSensTets))
print('NotCaSensTets: {}'.format(len(NotCaSensTets)))
print('CaSensTets:    {}'.format(len(CaSensTets)))


NotAzTris = list(set(surftris) - set(NotCaSensTris) - set(CaSensTris))

# get the furthest (from the origin) vertext to inject AP current
furthest = max([triBARYC[i][2] for i in range(len(triBARYC))])
furthest_tet = mesh.findTetByPoint([0,0,furthest])
furthest_tet_verts = mesh.getTet(furthest_tet)
furthest_vertex_id = np.argmin([mesh.getVertex(i)[2] for i in furthest_tet_verts])
furthest_vertex_id = furthest_tet_verts[furthest_vertex_id]

print('Furthest vertex ID: {}'.format(furthest_vertex_id))


# sort CS tets by distance from the origin:
tetBARYC, CStetrads = [], []
for CS_tet_id in CS_tet_IDs:
    baryc = mesh.getTetBarycenter(CS_tet_id)
    tetBARYC.append(baryc)
    r = math.sqrt(math.pow((baryc[0]-cbaryc[0]),2) \
                    + math.pow((baryc[1]-cbaryc[1]),2) \
                        + math.pow((baryc[2]-cbaryc[2]),2))
    CStetrads.append(r)
tmp = zip(CStetrads, CS_tet_IDs)
sorted_CS_tets = sorted(tmp, key=lambda x: x[0])



def draw_polygon_coll(ax, mesh, tris, sf=1, ln=1, pg=1, c=[0,0,1], alpha=0.1, linewidths=0.5, linestyles=':'):
    Pcoll = []
    for triID in tris:
        P = []
        for verID in mesh.getTri(triID):
            triVertex = tuple(mesh.getVertex(verID))
            P.append(triVertex)
        Pcoll.append(P)
    if ln==1:
        collection = Line3DCollection(Pcoll, colors=c, linewidths=linewidths, linestyles=linestyles)
        ax.add_collection3d(collection)
    if pg==1:
        collection = Poly3DCollection(Pcoll)
        collection.set_facecolor((c[0], c[1], c[2], alpha))
        
    ax.add_collection3d(collection)
    ax.set_xlim(0, SCALING_FACTOR)
    ax.set_ylim(0, SCALING_FACTOR)
    ax.set_zlim(0, SCALING_FACTOR)

fig = plt.figure(figsize=(8,7))
ax = Axes3D(fig)

all_ER_tris = meshctrl.findOverlapTris(mesh, CS_tet_IDs, ER_tet_IDs)
all_surf_tris = mesh.getSurfTris()
cent_tet = mesh.findTetByPoint([cbaryc[0], cbaryc[1], cbaryc[2] + 0.001*SCALING_FACTOR])
cent_tets = mesh.getTetTriNeighb(cent_tet)


mesh.getTetTriNeighb(1101)



# draw_polygon_coll(ax, mesh, cent_tets, sf=SCALING_FACTOR, pg=1, ln=1, c=[0,0,1], alpha=0.3, linewidths=0.5, linestyles=':')
draw_polygon_coll(ax, mesh, all_ER_tris, sf=SCALING_FACTOR, pg=1, ln=1, c=[1,0,0], alpha=0.1, linewidths=0.5, linestyles=':')
draw_polygon_coll(ax, mesh, NotCaSensTris, sf=SCALING_FACTOR, pg=1, ln=0, c=[0,0,1], alpha=0.2, linewidths=0.5, linestyles=':')
# draw_polygon_coll(ax, mesh, all_surf_tris, sf=SCALING_FACTOR, pg=0, ln=1, c=[0,1,0], alpha=0.1, linewidths=0.5, linestyles=':')
# draw_polygon_coll(ax, mesh, az_tet_tris, sf=SCALING_FACTOR, pg=0, ln=1, c=[1,0,1], alpha=0.9, linewidths=0.4, linestyles='-')
draw_polygon_coll(ax, mesh, CaSensTris, sf=SCALING_FACTOR, pg=1, ln=1, c=[1,0,0], alpha=0.9, linewidths=1, linestyles='-')
draw_polygon_coll(ax, mesh, NotAzTris, sf=SCALING_FACTOR, pg=0, ln=1, c=[0,0,0], alpha=0.4, linewidths=0.05, linestyles='-')

VoltmeterTetTris = mesh.getTetTriNeighb(1101)
draw_polygon_coll(ax, mesh, VoltmeterTetTris, sf=SCALING_FACTOR, pg=1, ln=1, c=[0,1,0], alpha=0.9, linewidths=1, linestyles='-')

furthest_vertex = mesh.getVertex(furthest_vertex_id)
ax.scatter(furthest_vertex[0], furthest_vertex[1], furthest_vertex[2], c='g', marker='d', s=30)
ax.set_title('MESH FILE: {}'.format(meshfile), fontsize=16, color='red', fontweight='bold')

ax.scatter(cbaryc[0], cbaryc[1], cbaryc[2], s=30, marker='d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')



model = smodel.Model()

vsys0 = smodel.Volsys('vsys0', model) # cytosol
vsys1 = smodel.Volsys('vsys1', model) # ER

surfsys0 =  smodel.Surfsys('surfsys0', model) # cytosol membrane
surfsys2 =  smodel.Surfsys('surfsys2', model) # ER memb
ssys =      smodel.Surfsys('ssys', model)     # AZ


########## Initial Membrane Potential #########
init_pot = -60e-3

vrange = [-400.0e-3, 400e-3, 1e-4]

########## BULK RESISTIVITY ##########
Ra = 235.7*1.0e-2

# Calcium channel
Ca = smodel.Spec('Ca', model)
# Ca_Ch = smodel.Spec('Ca_Ch', model)
# Ca_sens = smodel.Spec('Ca_sens', model)
# Ca_sens_b = smodel.Spec('Ca_sens_b', model)

Ca.setValence(2)
Ca_oconc = 2e-3

CaP_P = 0.5e-20      # permeability of a single channel (m3/s) HERE IS A PROBLEM (3.72 pS)
CaP_ro = 5 * 3.8e13  # density per square meter

CaPchan = smodel.Chan('CaPchan', model)
CaP_m0 = smodel.ChanState('CaP_m0', model, CaPchan)
CaP_m1 = smodel.ChanState('CaP_m1', model, CaPchan)
CaP_m2 = smodel.ChanState('CaP_m2', model, CaPchan)
CaP_m3 = smodel.ChanState('CaP_m3', model, CaPchan)

CaP_m0_p = 0.92402
CaP_m1_p = 0.073988
CaP_m2_p = 0.0019748
CaP_m3_p = 1.7569e-05

#Units (mV)
vhalfm = -29.458
cvm = 8.429

def minf_cap(V):
    #Units (mV)
    vhalfm = -29.458
    cvm = 8.429
    vshift = 0.0
    return (1.0/(1.0 + math.exp(-(V-vhalfm-vshift)/cvm)))

def tau_cap(V):
    vshift = 0.0
    if (V-vshift) >= -40:
        return (0.2702 + 1.1622 * math.exp(-(V+26.798-vshift)*(V+26.798-vshift)/164.19))
    else:
        return (0.6923 * math.exp((V-vshift)/1089.372))

def alpha_cap(V):
    return (minf_cap(V)/tau_cap(V))

def beta_cap(V):
    return ((1.0-minf_cap(V))/tau_cap(V))


CaPm0m1 = smodel.VDepSReac('CaPm0m1', ssys, slhs = [CaP_m0], srhs = [CaP_m1], k= lambda V: 1.0e3 *3.* alpha_cap(V*1.0e3), vrange=vrange)
CaPm1m2 = smodel.VDepSReac('CaPm1m2', ssys, slhs = [CaP_m1], srhs = [CaP_m2], k= lambda V: 1.0e3 *2.* alpha_cap(V*1.0e3), vrange=vrange)
CaPm2m3 = smodel.VDepSReac('CaPm2m3', ssys, slhs = [CaP_m2], srhs = [CaP_m3], k= lambda V: 1.0e3 *1.* alpha_cap(V*1.0e3), vrange=vrange)

CaPm3m2 = smodel.VDepSReac('CaPm3m2', ssys, slhs = [CaP_m3], srhs = [CaP_m2], k= lambda V: 1.0e3 *3.* beta_cap(V*1.0e3), vrange=vrange)
CaPm2m1 = smodel.VDepSReac('CaPm2m1', ssys, slhs = [CaP_m2], srhs = [CaP_m1], k= lambda V: 1.0e3 *2.* beta_cap(V*1.0e3), vrange=vrange)
CaPm1m0 = smodel.VDepSReac('CaPm1m0', ssys, slhs = [CaP_m1], srhs = [CaP_m0], k= lambda V: 1.0e3 *1.* beta_cap(V*1.0e3), vrange=vrange)

# current that will start for channels in the n4 conformational state:
OC_CaP = smodel.GHKcurr('OC_CaP', ssys, CaP_m3, Ca, virtual_oconc = Ca_oconc, computeflux = True)

#Set single channel permeability
OC_CaP.setP(CaP_P)



diff_Ca_cyt  = smodel.Diff('diff_Ca_cyt',  vsys0,    Ca,         DCST) # name, where, what, how fast
diff_Ca_ER  = smodel.Diff('diff_Ca_ER',    vsys1,    Ca,         DCST) # name, where, what, how fast

# diff_Ca_sens    = smodel.Diff('diff_Ca_sens', surfsys2, Ca_sens,  DCST) # name, where, what, how fast

# define compartments:
cyto_comp = stetmesh.TmComp('cyto_comp', mesh, CS_tet_IDs)
ER_comp = stetmesh.TmComp('ER_comp', mesh, ER_tet_IDs)

# get surf tris in comps:
CS_memb_tris = meshctrl.findSurfTrisInComp(mesh, cyto_comp)
ER_memb_tris = meshctrl.findOverlapTris(mesh, CS_tet_IDs, ER_tet_IDs)

# patches:
ER_surf = stetmesh.TmPatch('ER_surf', mesh, ER_memb_tris, icomp=ER_comp, ocomp=cyto_comp)
memb_surf = stetmesh.TmPatch('memb_surf', mesh, NotAzTris, icomp=cyto_comp, ocomp=None)
# AZ_surf     = stetmesh.TmPatch('AZ_surf',         mesh, az_tris,     icomp = cyto_comp)

# $-----$-----$-----$-----$-----$-----$-----$-----$-----$-----$-----$-----
# Create the patch and associate with surface system 'ssys'
patch = stetmesh.TmPatch('patch', mesh, NotCaSensTris, icomp = cyto_comp)

# Create the membrane across which the potential will be solved
membrane = stetmesh.Memb('membrane', mesh, [patch], opt_method = 1)
# $-----$-----$-----$-----$-----$-----$-----$-----$-----$-----$-----$-----

# add volume systems (with all their reactions and diff rules) to compartments:
cyto_comp.addVolsys('vsys0')
ER_comp.addVolsys('vsys1')
memb_surf.addSurfsys('surfsys0')
ER_surf.addSurfsys('surfsys2')
# $-----$-----$-----$-----$-----$-----$-----$-----$-----$-----$-----$-----
patch.addSurfsys('ssys')
# $-----$-----$-----$-----$-----$-----$-----$-----$-----$-----$-----$-----





# Reactions involving SERCA:
# # SERCA species:
# SERCA_X0Ca = smodel.Spec('SERCA_X0Ca', model)
# SERCA_X1Ca = smodel.Spec('SERCA_X1Ca', model)
# SERCA_X2Ca = smodel.Spec('SERCA_X2Ca', model)
# SERCA_Y2Ca = smodel.Spec('SERCA_Y2Ca', model)
# SERCA_Y1Ca = smodel.Spec('SERCA_Y1Ca', model)
# SERCA_Y0Ca = smodel.Spec('SERCA_Y0Ca', model)

# # on- and off-rates for SERCA:
# kx0x1 = 2e9 # !!!!!!!!!
# kx1x0 = 83.7 
# kx1x2 = 1e9 # !!!!!!!!!
# kx2x1 = 167.4
# kx2y2 = 0.6
# ky2x2 = 0.097
# ky2y1 = 60.04
# ky1y2 = 1e5
# ky1y0 = 30.02
# ky0y1 = 2e5
# ky0x0 = 0.4
# kx0y0 = 1.2e-3
# # when SERCA faces outward (into the cytosol):
# ser_x0x1 = smodel.SReac('ser_x0x1', surfsys2, olhs=[Ca], slhs=[SERCA_X0Ca], srhs=[SERCA_X1Ca], kcst=kx0x1)
# ser_x1x0 = smodel.SReac('ser_x1x0', surfsys2, slhs=[SERCA_X1Ca], olhs=[Ca], srhs=[SERCA_X0Ca], kcst=kx1x0)
# ser_x1x2 = smodel.SReac('ser_x1x2', surfsys2, olhs=[Ca], slhs=[SERCA_X1Ca], srhs=[SERCA_X2Ca], kcst=kx1x2)
# ser_x2x1 = smodel.SReac('ser_x2x1', surfsys2, slhs=[SERCA_X2Ca], olhs=[Ca], srhs=[SERCA_X1Ca], kcst=kx2x1)

# # we change conformation from inward-facing to outward-facing:
# ser_x2y2 = smodel.SReac('ser_x2y2', surfsys2, slhs=[SERCA_X2Ca], srhs=[SERCA_Y2Ca],            kcst=kx2y2)
# ser_y2x2 = smodel.SReac('ser_y2x2', surfsys2, slhs=[SERCA_Y2Ca], srhs=[SERCA_X2Ca],            kcst=ky2x2)

# # when SERCA faces inward (into the ER):
# ser_y2y1 = smodel.SReac('ser_y2y1', surfsys2, slhs=[SERCA_Y2Ca], srhs=[SERCA_Y1Ca], irhs=[Ca], kcst=ky2y1)
# ser_y1y2 = smodel.SReac('ser_y1y2', surfsys2, ilhs=[Ca], slhs=[SERCA_Y1Ca], srhs=[SERCA_Y2Ca], kcst=ky1y2)
# ser_y1y0 = smodel.SReac('ser_y1y0', surfsys2, slhs=[SERCA_Y1Ca], srhs=[SERCA_Y0Ca], irhs=[Ca], kcst=ky1y0)
# ser_y0y1 = smodel.SReac('ser_y0y1', surfsys2, ilhs=[Ca], slhs=[SERCA_Y0Ca], srhs=[SERCA_Y1Ca], kcst=ky0y1)

# # we change conformation from outward-facing to inward-facing:
# ser_y0x0 = smodel.SReac('ser_y0x0', surfsys2, slhs=[SERCA_Y0Ca], srhs=[SERCA_X0Ca]           , kcst=ky0x0)
# ser_x0y0 = smodel.SReac('ser_x0y0', surfsys2, slhs=[SERCA_X0Ca], srhs=[SERCA_Y0Ca]           , kcst=kx0y0)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////
# Ca + SERCA <->  Ca1SERCA +Ca <->  Ca2SERCA  ->  SERCA

DCST_MEM = 0.05e-12

SERCA = smodel.Spec('SERCA', model)
CaSERCA = smodel.Spec('CaSERCA',model)
Ca2SERCA = smodel.Spec('Ca2SERCA',model)

diff_SERCA = smodel.Diff('diff_SERCA', surfsys2, SERCA,  DCST_MEM)
diff_CaSERCA = smodel.Diff('diff_CaSERCA', surfsys2, CaSERCA,  DCST_MEM)
diff_Ca2SERCA = smodel.Diff('diff_Ca2SERCA', surfsys2, Ca2SERCA,  DCST_MEM)

Reac9  = smodel.SReac('Reac9',  surfsys2, olhs=[Ca], slhs=[SERCA], srhs=[CaSERCA], kcst=17147e6)
Reac10 = smodel.SReac('Reac10', surfsys2, slhs=[CaSERCA], orhs=[Ca], srhs=[SERCA], kcst=8426.3)
Reac11 = smodel.SReac('Reac11', surfsys2, olhs=[Ca], slhs=[CaSERCA], srhs=[Ca2SERCA], kcst=17147e6)
Reac12 = smodel.SReac('Reac12', surfsys2, slhs=[Ca2SERCA], orhs=[Ca], srhs=[CaSERCA], kcst=8426.3)
Reac13 = smodel.SReac('Reac13', surfsys2, slhs=[Ca2SERCA], srhs=[SERCA], irhs=[Ca,Ca], kcst=250)
# ////////////////////////////////////////////////////////////////////////////////////////////////////////

# PMCA species:
PMCA_P0 = smodel.Spec('PMCA_P0', model)
PMCA_P1 = smodel.Spec('PMCA_P1', model)

k1_pmca = 1.5e8
k2_pmca = 15
k3_pmca = 12
kl_pmca = 4.3


PMCA_P0P1 = smodel.SReac('PMCA_P0P1',       surfsys0, ilhs=[Ca], slhs=[PMCA_P0], srhs=[PMCA_P1], kcst=k1_pmca)
PMCA_P1P0 = smodel.SReac('PMCA_P1P0',       surfsys0, slhs=[PMCA_P1], srhs=[PMCA_P0], ilhs=[Ca], kcst=k2_pmca)
PMCA_P1P0ext = smodel.SReac('PMCA_P1P0ext', surfsys0, slhs=[PMCA_P1], srhs=[PMCA_P0],            kcst=k3_pmca)
PMCA_leak = smodel.SReac('PMCA_leak',       surfsys0, slhs=[PMCA_P0], srhs=[PMCA_P0], irhs=[Ca], kcst=kl_pmca)


# ////////////////////////////////////////////////////////////////////////////////////////////////////////
PV = smodel.Spec('PV', model) 
PV_Ca = smodel.Spec('PV_Ca', model)
PV_2Ca = smodel.Spec('PV_2Ca', model)

kreac_f_PV_Ca = smodel.Reac('kreac_f_PV_Ca', vsys0, lhs=[PV, Ca], rhs=[PV_Ca], kcst=107e6)
kreac_b_PV_Ca = smodel.Reac('kreac_b_PV_Ca', vsys0, lhs=[PV_Ca], rhs=[PV, Ca], kcst=0.95)

kreac_f_PV_2Ca = smodel.Reac('kreac_f_PV_2Ca', vsys0, lhs=[PV_Ca, Ca], rhs=[PV_2Ca], kcst=107e6)
kreac_b_PV_2Ca = smodel.Reac('kreac_b_PV_2Ca', vsys0, lhs=[PV_2Ca], rhs=[PV_Ca, Ca], kcst=0.95)

# ////////////////////////////////////////////////////////////////////////////////////////////////////////
CBhi = smodel.Spec('CBhi', model)         # CALBINDIN 1 (D-28K) (we consider the 2 (hi-aff) : 2 (lo-aff) scenario)
CBhi_Ca = smodel.Spec('CBhi_Ca', model)   # Binding Kinetics of Calbindin-D28k Determined by Flash Photolysis of Caged Ca2 (Nagerl 2000)
CBhi_2Ca = smodel.Spec('CBhi_2Ca', model)
CBlo = smodel.Spec('CBlo', model)
CBlo_Ca = smodel.Spec('CBlo_Ca', model)
CBlo_2Ca = smodel.Spec('CBlo_2Ca', model)


kreac_f_CBhi_Ca = smodel.Reac('kreac_f_CBhi_Ca', vsys0, lhs = [CBhi, Ca], rhs = [CBhi_Ca], kcst = 1.1e7)
kreac_b_CBhi_Ca = smodel.Reac('kreac_b_CBhi_Ca', vsys0, lhs = [CBhi_Ca], rhs = [CBhi, Ca], kcst = 2.607)

kreac_f_CBhi_2Ca = smodel.Reac('kreac_f_CBhi_2Ca', vsys0, lhs = [CBhi_Ca, Ca], rhs = [CBhi_2Ca], kcst = 1.1e7)
kreac_b_CBhi_2Ca = smodel.Reac('kreac_b_CBhi_2Ca', vsys0, lhs = [CBhi_2Ca], rhs = [CBhi_Ca, Ca], kcst = 2.607)

kreac_f_CBlo_Ca = smodel.Reac('kreac_f_CBlo_Ca', vsys0, lhs = [CBlo, Ca], rhs = [CBlo_Ca], kcst = 8.7e7)
kreac_b_CBlo_Ca = smodel.Reac('kreac_b_CBlo_Ca', vsys0, lhs = [CBlo_Ca], rhs = [CBlo, Ca], kcst = 35.76)

kreac_f_CBlo_2Ca = smodel.Reac('kreac_f_CBlo_2Ca', vsys0, lhs = [CBlo_Ca, Ca], rhs = [CBlo_2Ca], kcst = 8.7e7)
kreac_b_CBlo_2Ca = smodel.Reac('kreac_b_CBlo_2Ca', vsys0, lhs = [CBlo_2Ca], rhs = [CBlo_Ca, Ca], kcst = 35.76)


# ////////////////////////////////////////////////////////////////////////////////////////////////////////
Eggerman2002 = True

CaM_NtNt = smodel.Spec('CaM_NtNt', model)
CaM_NtNr = smodel.Spec('CaM_NtNr', model)
CaM_NrNr = smodel.Spec('CaM_NrNr', model)

CaM_CtCt = smodel.Spec('CaM_CtCt', model)
CaM_CtCr = smodel.Spec('CaM_CtCr', model)
CaM_CrCr = smodel.Spec('CaM_CrCr', model)

if Eggerman2002==True:
    KonT_N  = 7.7e8
    KoffT_N = 1.6e5
    KonR_N  = 3.2e10
    KoffR_N = 2.2e4

    KonT_C  = 8.4e7
    KoffT_C = 2.6e3
    KonR_C  = 2.5e7
    KoffR_C = 6.5
else:
    KonT_N  = 8.9e8
    KoffT_N = 5.2e5
    KonR_N  = 10.5e10
    KoffR_N = 4.3e4

    KonT_C  = 7.9e7
    KoffT_C = 3.4e3
    KonR_C  = 7.4e7
    KoffR_C = 1.2


kreac_f_CaM_NtNt = smodel.Reac('kreac_f_CaM_NtNt', vsys0, lhs = [CaM_NtNt, Ca], rhs = [CaM_NtNr], kcst = 2 * KonT_N)
kreac_b_CaM_NtNt = smodel.Reac('kreac_b_CaM_NtNt', vsys0, lhs = [CaM_NtNr], rhs = [CaM_NtNt, Ca], kcst = KoffT_N)

kreac_f_CaM_NrNr = smodel.Reac('kreac_f_CaM_NrNr', vsys0, lhs = [CaM_NtNr, Ca], rhs = [CaM_NrNr], kcst = KonR_N)
kreac_b_CaM_NrNr = smodel.Reac('kreac_b_CaM_NrNr', vsys0, lhs = [CaM_NrNr], rhs = [CaM_NtNr, Ca], kcst = 2 * KoffR_N)

kreac_f_CaM_CtCt = smodel.Reac('kreac_f_CaM_CtCt', vsys0, lhs = [CaM_CtCt, Ca], rhs = [CaM_CtCr], kcst = 2 * KonT_C)
kreac_b_CaM_CtCt = smodel.Reac('kreac_b_CaM_CtCt', vsys0, lhs = [CaM_CtCr], rhs = [CaM_CtCt, Ca], kcst = KoffT_C)

kreac_f_CaM_CrCr = smodel.Reac('kreac_f_CaM_CrCr', vsys0, lhs = [CaM_CtCr, Ca], rhs = [CaM_CrCr], kcst = KonR_C)
kreac_b_CaM_CrCr = smodel.Reac('kreac_b_CaM_CrCr', vsys0, lhs = [CaM_CrCr], rhs = [CaM_CtCr, Ca], kcst = 2 * KoffR_C)





# ////////////////////////////////////////////////////////////////////////////////////////////////////////
CRTT     = smodel.Spec('CRTT', model)     # pairs of cooperative binding sites MULTIPLY BY TWO to get 4
CRTR_Ca  = smodel.Spec('CRTR_Ca', model)
CRRR_2Ca = smodel.Spec('CRRR_2Ca', model)
CRind     = smodel.Spec('CRind', model)
CRind_Ca  = smodel.Spec('CRind_Ca', model) # independent binding site

kon_T  = 1.8e6
koff_T = 53
kon_R  = 3.1e8
koff_R = 20
kon_ind  = 7.3e6
koff_ind = 252

# pair 1 (you MUST multiply its concentration by 2) because WE'VE GOT TWO PAIRS OF COOPERATIVE CA2+ BINDING SITES
kreac_f_CRTT1_Ca = smodel.Reac('kreac_f_CRTT1_Ca', vsys0, lhs = [CRTT, Ca], rhs = [CRTR_Ca], kcst=2*kon_T) # two domains are free, so we multiply the association rate by 2
kreac_b_CRTT1_Ca = smodel.Reac('kreac_b_CRTT1_Ca', vsys0, lhs = [CRTR_Ca], rhs = [CRTT, Ca], kcst=koff_T)
kreac_f_CRRR1_2Ca = smodel.Reac('kreac_f_CRRR1_2Ca', vsys0, lhs = [CRTR_Ca, Ca], rhs = [CRRR_2Ca], kcst=kon_T) 
kreac_b_CRRR1_2Ca = smodel.Reac('kreac_b_CRRR1_2Ca', vsys0, lhs = [CRRR_2Ca], rhs = [CRTR_Ca, Ca], kcst = 2*koff_R) # two domains are occupied, so we multiply the dissociation rate by 2

# independent Ca2+ binding site:
kreac_f_CRind_Ca = smodel.Reac('kreac_f_CRind_Ca', vsys0, lhs = [CRind, Ca], rhs = [CRind_Ca], kcst=kon_ind)
kreac_b_CRind_Ca = smodel.Reac('kreac_b_CRind_Ca', vsys0, lhs = [CRind_Ca], rhs = [CRind, Ca], kcst=koff_ind)

print('Number of channels in the AZ: {:.1f}'.format(CaP_ro * NotCaSensTris_area))

CytoCompSpecs = ['Ca', 'CRTT', 'CRTR_Ca', 'CRRR_2Ca', 'CRind', 'CRind_Ca', 'CaM_NtNt', 'CaM_NtNr', 'CaM_NrNr', 'CaM_CtCt', 'CaM_CtCr', 'CaM_CrCr', 'CBhi', 'CBlo', 'CBhi_Ca', 'CBhi_2Ca', 'CBlo_Ca', 'CBlo_2Ca', 'PV', 'PV_Ca', 'PV_2Ca']
a = list(zip(CytoCompSpecs, ['cyto_comp'] * len(CytoCompSpecs), ['comp'] * len(CytoCompSpecs)))
AzPatchSpecs = ['CaP_m0', 'CaP_m1', 'CaP_m2', 'CaP_m3']
b = list(zip(AzPatchSpecs, ['patch'] * len(AzPatchSpecs), ['patch'] * len(AzPatchSpecs)))
ErSurfSpecs = ['SERCA', 'CaSERCA', 'Ca2SERCA']
c = list(zip(ErSurfSpecs, ['ER_surf'] * len(ErSurfSpecs), ['patch'] * len(ErSurfSpecs)))
ErCompSpecs = ['Ca']
d = list(zip(ErCompSpecs, ['ER_comp'] * len(ErCompSpecs), ['comp'] * len(ErCompSpecs)))
specs = a + b + c + d
specs



def ResetSim (sim, model, mesh, rng):
    SERCA_ro =      1000*1e12 # Bartol 2015. Computational reconstitution...
    init_memb_pot = -60e-3
    memb_cap =      1.0e-2
    memb_resist =   1.0
    
    sim.reset()

    # total CB in rat HC: 1.98e-6
    sim.setCompConc('cyto_comp', 'CBhi', 0.99e-6) # i.e. 1/2 of total CB molarity (1.98*10e-6)
    sim.setCompConc('cyto_comp', 'CBlo', 0.99e-6) # i.e. 1/2 of total CB molarity (1.98*10e-6)

    # total CaM in rat HC: 57.82e-6
    sim.setCompConc('cyto_comp', 'CaM_NtNt', 0.5 * 57.82e-6) # i.e. 1/2 of total CaM molarity
    sim.setCompConc('cyto_comp', 'CaM_CtCt', 0.5 * 57.82e-6) # i.e. 1/2 of total CaM molarity

    # total CR in rat HC: 2.47e-6
    # WE MODEL TWO IDENTICAL PAIRS OF COOPERATIVE BINDING SITES AS ONE SPECIES.
    # SO WE SET ITS CONCENTRATION TO 4/5 OF THE TOTAL CONCENTRATION OF CR. THE REMAINING 1/5 IS THE INDEPENDENT SITE/
    sim.setCompConc('cyto_comp', 'CRTT', 0.1976e-6)   # 4/5 of total CR concentration (2 pairs of cooperative sites)
    sim.setCompConc('cyto_comp', 'CRind', 0.494e-6)  # 1/5 of total CR
    
    # total PV in rat HC: 4.55e-6
    sim.setCompConc('cyto_comp', 'PV', 4.55e-6)

    SERCA_count = sim.getPatchArea('ER_surf') * SERCA_ro
    sim.setPatchCount('ER_surf', 'SERCA', SERCA_count)
#     sim.setPatchCount('memb_surf', 'PMCA_P0', 828)
#     sim.setPatchCount('memb_surf', 'NCX_P0', 11)    
    
    surfarea = sim.getPatchArea('patch')
    sim.setPatchCount('patch', 'CaP_m0', round(CaP_ro*surfarea*CaP_m0_p))
    sim.setPatchCount('patch', 'CaP_m1', round(CaP_ro*surfarea*CaP_m1_p))
    sim.setPatchCount('patch', 'CaP_m2', round(CaP_ro*surfarea*CaP_m2_p))
    sim.setPatchCount('patch', 'CaP_m3', round(CaP_ro*surfarea*CaP_m3_p))

    # Set dt for membrane potential calculation to 0.01ms
    sim.setEfieldDT(0.0003)

    # Initialise potential to -65mV
    sim.setMembPotential('membrane', init_memb_pot)

    # Set capacitance of the membrane to 1 uF/cm^2 = 0.01 F/m^2
    sim.setMembCapac('membrane', memb_cap)

    # Set resistivity of the conduction volume to 100 ohm.cm = 1 ohm.meter
    sim.setMembVolRes('membrane', memb_resist)
    
    return sim

def getAmpVolt(sim, NotCaSensTris):
    I, V = [], []
    for i in NotCaSensTris:
        I.append(sim.getTriGHKI(i, 'OC_CaP'))
        V = [sim.getTriV(i) for i in NotCaSensTris]
    return sum(I), np.mean(V)

def set_clamp_current(sim, Iclamp, furthest_vertex_id):
    sim.setVertIClamp(furthest_vertex_id, Iclamp)
    return sim

def PlotMeanSpec(spec, specs, res):
    idx = [i for i,j in enumerate(specs) if j[0]==spec]
    for i in idx:
        plt.plot(np.mean(res[:,i,:], axis=0), label='{} in {}'.format(specs[i][0], specs[i][1]))
    plt.legend()


CA_FLUX_ONSET =  0.020
CA_FLUX_OFFSET = 0.021
CA_FLUSH =       0.030

NITER = 100                 # The number of iterations to run
DT = 0.0003               # The data collection time increment (s)
T = 0.04                  # The simulation endtime (s)
OC_CaP.setP(CaP_P)        # Set single channel permeability

rng = srng.create('mt19937', 512)
rng.initialize(2903)

sim = solvmod.Tetexact(model, mesh, rng, True)
pbarlen = 50              # initialize a progressbar:
tpnts = numpy.arange(0.0, T, DT)
ntpnts = tpnts.shape[0]
res = np.zeros((NITER, len(specs)+2, ntpnts))

for i in range(NITER):
	tt = time.time()
	sim = ResetSim(sim, model, mesh, rng)
	printProgressBar(0, NITER, prefix = 'Progress:', suffix = 'Complete', length = pbarlen)
	for j in range(ntpnts):
		if tpnts[j] > CA_FLUX_ONSET and tpnts[j] < CA_FLUX_OFFSET:
			sim.setMembPotential('membrane', 25e-3)
			# set_clamp_current(sim, 10e-14, furthest_vertex_id)
			# note = colored('CLAMPING', 'red')
			note = 'CLAMPING'
		else:
			# set_clamp_current(sim, 0.0, furthest_vertex_id)
			sim.setMembPotential('membrane', -60e-3)
			note = 'NOT clamping'
			# note = colored('NOT clamping', 'blue')
		if tpnts[j] > CA_FLUSH:
			sim.setCompCount('cyto_comp', 'Ca', 0)
			# note = colored('Ca removed', 'green')
			note = 'Ca removed'
		sim.run(tpnts[j])
		
		for spec_id in range(len(specs)):
			if specs[spec_id][2]=='comp':
				res[i,spec_id,j] = sim.getCompCount(specs[spec_id][1], specs[spec_id][0])
			else:
				res[i,spec_id,j] = sim.getPatchCount(specs[spec_id][1], specs[spec_id][0])
		I, V = getAmpVolt(sim, NotCaSensTris)
		res[i,len(specs),j] = I
		res[i,len(specs)+1,j] = V
		printProgressBar(j + 1, ntpnts, prefix='Progress:', suffix='Complete \t' + note, length=pbarlen)
	print('Time per iteration: {:.4f} \t sec.'.format(time.time() - tt))
	f = open("time_dump.txt", "a")
	f.writelines('Time per iteration: {:.4f} \t sec. \n'.format(time.time() - tt))
	f.close()

plt.figure(figsize=(17,12))

plt.subplot(4,3,1)
plt.plot(np.mean(res[:,-2,:], axis=0), label='{} in {}'.format('Current', 'membrane'))
plt.legend()

plt.subplot(4,3,2)
plt.plot(np.mean(res[:,-1,:], axis=0), label='{} in {}'.format('Voltage', 'membrane'))
plt.legend()

plt.subplot(4,3,3)
PlotMeanSpec('PV', specs, res)

plt.subplot(4,3,4)
PlotMeanSpec('CBlo', specs, res)
PlotMeanSpec('CBhi', specs, res)

plt.subplot(4,3,5)
PlotMeanSpec('CBlo_Ca', specs, res)
PlotMeanSpec('CBlo_2Ca', specs, res)
PlotMeanSpec('CBhi_Ca', specs, res)
PlotMeanSpec('CBhi_2Ca', specs, res)

plt.subplot(4,3,6)
PlotMeanSpec('CaM_NtNt', specs, res)
PlotMeanSpec('CaM_CtCt', specs, res)

plt.subplot(4,3,7)
PlotMeanSpec('CaM_NtNr', specs, res)
PlotMeanSpec('CaM_NrNr', specs, res)
PlotMeanSpec('CaM_CtCr', specs, res)
PlotMeanSpec('CaM_CrCr', specs, res)

plt.subplot(4,3,8)
PlotMeanSpec('CRTT', specs, res)
PlotMeanSpec('CRind', specs, res)

plt.subplot(4,3,9)
PlotMeanSpec('Ca', specs, res)

plt.subplot(4,3,10)
PlotMeanSpec('CRTR_Ca', specs, res)
PlotMeanSpec('CRRR_2Ca', specs, res)

plt.subplot(4,3,11)
PlotMeanSpec('PV_Ca', specs, res)
PlotMeanSpec('PV_2Ca', specs, res)

plt.subplot(4,3,12)
PlotMeanSpec('CaP_m0', specs, res)
PlotMeanSpec('CaP_m1', specs, res)
PlotMeanSpec('CaP_m2', specs, res)
PlotMeanSpec('CaP_m3', specs, res)

plt.savefig('fig.png')


