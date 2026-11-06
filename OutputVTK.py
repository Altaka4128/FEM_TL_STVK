import numpy as np
from pyvtk import *

class OutputVTK:

    def __init__(self, nodes, elements, vecDispList, vecRFList, elemOutputDataList, incNum):
        # インスタンス変数を定義する
        self.nodeDof            = 3
        self.nodes              = nodes
        self.elements           = elements
        self.vecDispList        = vecDispList
        self.vecRFList          = vecRFList
        self.elemOutputDataList = elemOutputDataList
        self.incNum             = incNum
 
    def output(self, filePath):

        # 初期データを作成する
        nodeData = []
        for node in self.nodes:
            nodeData.append([node.x, node.y, node.z])
        elemData = []
        for elem in self.elements:
            elemNodeNo = [elem.nodes[i].no - 1 for i in range(elem.nodeNum)]
            elemData.append(elemNodeNo)
        unStructure = UnstructuredGrid(points=nodeData, hexahedron=elemData)
        vecZeroData = np.zeros([self.nodeDof * len(self.nodes)])
        vecDispData = [vecZeroData[j:j + self.nodeDof] for j in range(0, len(vecZeroData), self.nodeDof)]
        vecRFData = [vecZeroData[j:j + self.nodeDof] for j in range(0, len(vecZeroData), self.nodeDof)]
        pointData = PointData(Vectors(vecDispData, name="Displacement"), Vectors(vecRFData, name="Reaction Force"))
        misesData = [0.0 for j in range(len(self.elements))]
        cellData  = CellData(Scalars(misesData, name="Mises"))
        vtk = VtkData(unStructure, pointData, cellData)
        vtk.tofile(filePath + "_"  + f"{0:04}", "binary")

        # 出力データを作成する
        for i in range(self.incNum):
            vecDispData = [self.vecDispList[i][j:j + self.nodeDof] for j in range(0, len(self.vecDispList[i]), self.nodeDof)]
            vecRFData   = [self.vecRFList[i][j:j + self.nodeDof] for j in range(0, len(self.vecRFList[i]), self.nodeDof)]
            pointData = PointData(Vectors(vecDispData, name="Displacement"), Vectors(vecRFData, name="Reaction Force"))
            misesData = [self.elemOutputDataList[i][j].centMises for j in range(len(self.elements))]
            cellData  = CellData(Scalars(misesData, name="Mises"))
            vtk = VtkData(unStructure, pointData, cellData)
            vtk.tofile(filePath + "_" + f"{i+1:04}", "binary")