class MakeFEM2dData:
    # コンストラクタ
    # readData : ReadFileクラスで読み込んだデータ(ReadFile型)
    def __init__(self, readData = None):

        # インスタンス変数を定義する
        self.nodes = []      # 全節点のリスト(Node2d型のリスト)
        self.elements = []   # 全要素のリスト(要素型のリスト)
        self.bound = None    # 境界条件(Boundary2d型)
        self.incNum = 1      # インクリメント数
        self.itrNum = 100    # ニュートン法のイテレータの上限
        self.rn = 0.005      # ニュートン法の残差力の収束判定のパラメータ
        self.cn = 0.01       # ニュートン法の変位の収束判定のパラメータ
        
        # 引数をチェックする
        if readData == None:
            return

        # 節点を格納する
        self.nodes = readData.nodes.nodes

        # 要素のリストを作成する
        nodeNoList = {}
        i = 0
        for node in self.nodes:
            nodeNoList[node.no] = i
            i += 1
        sortSecSets = self.sortSecSets(readData.sectionSets)
        elements = []
        for secSet in sortSecSets:
            for tmpElem in secSet.elemSet.elements:
                if secSet.getElementType() == "d2cps3":
                    elemNodes = [self.nodes[nodeNoList[tmpElem.nodesNo[0]]], 
                                 self.nodes[nodeNoList[tmpElem.nodesNo[1]]], 
                                 self.nodes[nodeNoList[tmpElem.nodesNo[2]]]]
                    #element = d2cps3(tmpElem.no, elemNodes, secSet)
                    #elements.append(element)
                if secSet.getElementType() == "d2cps4":
                    elemNodes = [self.nodes[nodeNoList[tmpElem.nodesNo[0]]], 
                                 self.nodes[nodeNoList[tmpElem.nodesNo[1]]], 
                                 self.nodes[nodeNoList[tmpElem.nodesNo[2]]],
                                 self.nodes[nodeNoList[tmpElem.nodesNo[3]]]]
                    #element = d2cps4(tmpElem.no, elemNodes, secSet)
                    #elements.append(element)
        self.elements = elements


        # 境界条件を格納する
        self.bound = readData.boundary

    # 要素が三角形、四辺形、四面体、六面体、三角柱、ピラミッドの順になるように並び替える
    def sortSecSets(self, secSets):
        
        # 要素の節点検索用のリストを定義する
        node3Set = ["d2cps3"]
        node4Set = ["d2cps4"]

        # 要素を節点数毎に分類する
        triElemSecSets = []
        quadElemSecSets = []
        for secSet in secSets:
            if secSet.getElementType() in node3Set:
                triElemSecSets.append(secSet)
            if secSet.getElementType() in node4Set:
                quadElemSecSets.append(secSet)
        
        # 要素を種類でソートする
        sortSecSets = []
        sortSecSets.append(triElemSecSets)
        sortSecSets.append(quadElemSecSets)
        sortSecSets = sum(sortSecSets, [])   # 要素を1次元リストにする

        return sortSecSets

    # 形状データをVTK形式で出力する
    def outputShapeVTK(self, filePath):
        outVtk = OutputVTK()
        outVtk.setNodes2d(self.nodes)
        outVtk.setElements(self.elements)
        outVtk.output(filePath)
