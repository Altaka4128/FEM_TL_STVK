import copy
import time
from Node import Node
from MaterialSet import MaterialSet
from StVKMaterial import StVKMaterial
from Boundary import Boundary
from FEM import FEM
from C3D8 import C3D8

# メインの処理
def main():
    # ============================
    # メッシュ設定
    # ============================
    nx = 2     # x方向要素数
    ny = 2     # y方向要素数
    nz = 200   # z方向要素数
    lx = 10.0
    ly = 10.0
    lz = 500.0
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz
    # ============================
    # 節点生成
    # ============================
    node_dict = {}
    node_id = 1
    for ix in range(nx + 1):
        x = -lx / 2 + ix * dx
        for iy in range(ny + 1):
            y = -ly / 2 + iy * dy
            for iz in range(nz + 1):
                z = lz - iz * dz
                node_dict[node_id] = Node(node_id,x,y,z)
                node_id += 1
    nodes = [node_dict[i] for i in range(1, node_id)]
    # ============================
    # 節点番号取得関数
    # ============================
    nodes_per_y = ny + 1
    nodes_per_z = nz + 1
    def nid(ix, iy, iz):
        return (ix * nodes_per_y * nodes_per_z + iy * nodes_per_z + iz+ 1)
    # ============================
    # 材料
    # ============================
    young = 210000.0
    poisson = 0.0
    mat = StVKMaterial(young,poisson)
    matSet = MaterialSet("None",material=mat)
    # ============================
    # 要素生成
    # ============================
    elems = []
    elem_id = 1
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                n1 = node_dict[nid(ix  , iy  , iz+1)]
                n2 = node_dict[nid(ix+1, iy  , iz+1)]
                n3 = node_dict[nid(ix+1, iy+1, iz+1)]
                n4 = node_dict[nid(ix  , iy+1, iz+1)]
                n5 = node_dict[nid(ix  , iy  , iz)]
                n6 = node_dict[nid(ix+1, iy  , iz)]
                n7 = node_dict[nid(ix+1, iy+1, iz)]
                n8 = node_dict[nid(ix  , iy+1, iz)]
                elems.append(C3D8(elem_id,[n1, n2, n3, n4,n5, n6, n7, n8],matSet))
                elem_id += 1
    # ============================
    # 境界条件
    # ============================
    bound = Boundary(len(nodes))
    # 固定端(z=0)
    for ix in range(nx + 1):
        for iy in range(ny + 1):
            node = node_dict[nid(ix, iy, nz)]
            bound.addSPC(node.no,0.0,0.0,0.0)

    # 自由端中央
    bound1 = copy.deepcopy(bound)
    bound2 = copy.deepcopy(bound)
    tip = node_dict[nid(nx // 2,ny // 2,0)]
    bound1.addForce(tip.no,10.0,0.0,0.0)
    bound2.addForce(tip.no,5000.0,0.0,0.0)
    # ============================
    # FEM解析
    # ============================
    fem1 = FEM(nodes,elems,bound1,10)
    start = time.perf_counter() #計測開始
    fem1.impAnalysis()
    end = time.perf_counter() #計測終了
    print("解析時間 : " + '{:.2f}'.format((end-start)/60) + "分")
    # ============================
    # 結果出力
    # ============================
    fem1.outputTxt("test_10N")
    fem1.outputVTK("test_10N")
    # ============================
    # FEM解析
    # ============================
    fem2 = FEM(nodes,elems,bound2,10)
    start = time.perf_counter() #計測開始
    fem2.impAnalysis()
    end = time.perf_counter() #計測終了
    print("解析時間 : " + '{:.2f}'.format((end-start)/60) + "分")
    # ============================
    # 結果出力
    # ============================
    fem2.outputTxt("test_5000N")
    fem2.outputVTK("test_5000N")

if __name__ == '__main__':
    main()
