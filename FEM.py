import numpy as np
import numpy.linalg as LA
from scipy.sparse import lil_matrix, coo_matrix
from scipy.sparse.linalg import spsolve
from OutputVTK import OutputVTK

class FEM:
    # コンストラクタ
    # nodes    : 節点は1から始まる順番で並んでいる前提(Node型のリスト)
    # elements : 要素は種類ごとにソートされている前提(C3D8型のリスト)
    # bound    : 境界条件(d2Boundary型)
    # incNum   : インクリメント数
    def __init__(self, nodes, elements, bound, incNum):

        # インスタンス変数を定義する
        self.nodeDof = 3           # 節点の自由度
        self.nodes = nodes         # 節点は1から始まる順番で並んでいる前提(リスト)
        self.elements = elements   # 要素は種類ごとにソートされている前提(リスト)
        self.bound = bound         # 境界条件(Boundary型)
        self.incNum = incNum       # インクリメント数
        self.itrNum = 100          # ニュートン法のイテレータの上限
        self.rn = 0.005            # ニュートン法の残差力の収束判定のパラメータ
        #self.cn = 0.01             # ニュートン法の変位の収束判定のパラメータ
        self.cn = 0.001            # ニュートン法の変位の収束判定のパラメータ

    # 陰解法で解析を行う
    def impAnalysis(self):

        self.vecDispList = []          # インクリメント毎の変位ベクトルのリスト(np.array型のリスト)
        self.vecRFList = []            # インクリメント毎の反力ベクトルのリスト(np.array型のリスト)
        self.elemOutputDataList = []   # インクリメント毎の要素出力のリスト(makeOutputData型のリストのリスト)

        # 荷重をインクリメント毎に分割する
        vecfList = []
        for i in range(self.incNum):
            vecfList.append(self.makeForceVector() * (i + 1) / self.incNum)

        # ステータス表示用
        print("------------")
        print("Status")
        print("------------")
        print("INC".rjust(6) + "ITR".rjust(6)) 

        # ニュートン法により変位を求める
        vecDisp = np.zeros(len(self.nodes) * self.nodeDof)   # 全節点の変位ベクトル
        vecR = np.zeros(len(self.nodes) * self.nodeDof)      # 残差力ベクトル
        for i in range(self.incNum):
            vecDispFirst = vecDisp.copy()   # 初期の全節点の変位ベクトル
            vecf = vecfList[i]              # i+1番インクリメントの荷重
            vecBoundDisp = self.bound.makeDispVector()

            # 境界条件を考慮しないインクリメント初期の残差力ベクトルRを作成する
            if i == 0:
                vecR = vecfList[0]
            else:
                vecR = vecfList[i] - vecfList[i - 1]

            # 収束演算を行う
            matKt = self.makeKtmatrix()   # 接線剛性マトリクスKt(CSR)を作成する
            matKtc, vecRc = self.setBoundCondition(matKt, vecR, vecBoundDisp, vecDisp)   # 境界条件を考慮したKtcマトリクス、Rcベクトルを作成する
            for j in range(self.itrNum):

                # 変位ベクトルを計算する
                vecd = spsolve(matKtc, vecRc)
                if not np.isfinite(vecd).all():
                    raise ValueError("有限要素法の計算に失敗しました。")
                vecDisp += vecd

                # 全ての要素内の変数を更新する
                self.updateElements(vecDisp, i+1)

                # 新たな残差力ベクトルRを求める
                vecQ = self.makeVecQ()
                vecR = vecf - vecQ

                # 新たな境界条件を考慮したRcベクトルを作成する
                matKt = self.makeKtmatrix()
                matKtc, vecRc = self.setBoundCondition(matKt, vecR, vecBoundDisp, vecDisp)

                # 時間平均力を計算する
                aveForce = 0.0
                cnt = len(self.nodes) * self.nodeDof
                for k in range(len(vecQ)):
                    aveForce += np.abs(vecQ[k])
                for k in range(len(vecf)):
                    if not vecf[k] == 0.0:
                        aveForce += np.abs(vecf[k])
                        cnt+= 1
                aveForce = aveForce / cnt

                # 収束判定を行う
                if np.allclose(vecRc, 0.0):
                    break
                if np.isclose(LA.norm(vecd), 0.0):
                    break
                dispRate = LA.norm(vecd) / LA.norm(vecDisp - vecDispFirst)
                ResiForceRate = np.abs(vecRc).max() / aveForce
                if dispRate < self.cn and ResiForceRate < self.rn:
                    break

                # 収束できなかった場合
                if j == (self.itrNum - 1):
                    raise ValueError("収束しませんでした。")
        
            # インクリメントの最終的な変位べクトルを格納する
            self.vecDispList.append(vecDisp.copy())

            # 変位ベクトルから要素の出力データを計算する
            elemOutputDatas = []
            for elem in self.elements:
                elemOutputData = elem.makeOutputData()
                elemOutputDatas.append(elemOutputData)        
            self.elemOutputDataList.append(elemOutputDatas)

            # 節点反力を計算する
            vecRF = np.array(vecQ - vecf).flatten()

            # インクリメントの最終的な節点反力を格納する
            self.vecRFList.append(vecRF)    

            # ステータスを表示する
            print(str(i+1).rjust(6) + str(j+1).rjust(6))

        # 完了したことを表示する
        print("The analysis has completed") 
    
    # 接線剛性マトリクスKtを作成する(CSR形式)
    def makeKtmatrix(self):

        nDof = len(self.nodes) * self.nodeDof
        elemNum = len(self.elements)
        nnz = elemNum * 24 * 24
        rows = np.empty(nnz, dtype=np.int32)
        cols = np.empty(nnz, dtype=np.int32)
        data = np.empty(nnz)

        p = 0
        for elem in self.elements:
            
            # ketマトリクスを計算する
            matKet = elem.makeKetmatrix()

            dofs = np.empty(24, dtype=np.int32)

            for i in range(8):
                base = (elem.nodes[i].no - 1) * 3
                dofs[3 * i]     = base
                dofs[3 * i + 1] = base + 1
                dofs[3 * i + 2] = base + 2

            for i in range(24):
                for j in range(24):
                    rows[p] = dofs[i]
                    cols[p] = dofs[j]
                    data[p] = matKet[i,j]
                    p += 1

        return coo_matrix((data,(rows,cols)),shape=(nDof,nDof)).tocsr()

    # 節点に負荷する荷重、等価節点力を考慮した荷重ベクトルを作成する
    def makeForceVector(self):

        # 節点に負荷する荷重ベクトルを作成する
        vecCondiForce = self.bound.makeForceVector()
        vecf = vecCondiForce

        return vecf


    # Kマトリクス、荷重ベクトルに境界条件を考慮する
    # matKt        : 接線剛性マトリクス
    # vecR         : 残差力ベクトル
    # vecBoundDisp : 節点の境界条件の変位ベクトル
    # vecDisp      : 全節点の変位ベクトル(np.array型)
    def setBoundCondition(self, matKt, vecR, vecBoundDisp, vecDisp):

        matKtc = matKt.tolil().copy()
        vecRc = np.copy(vecR)

        # 単点拘束条件を考慮したKマトリクス、荷重ベクトルを作成する
        for i in range(len(vecBoundDisp)):
            if not vecBoundDisp[i] == None:
                # Kマトリクスからi列を抽出する
                vecx = np.array(matKt[:, i].todense()).flatten()

                # 変位ベクトルi列の影響を荷重ベクトルに適用する
                vecRc = vecRc - (vecBoundDisp[i] - vecDisp[i]) * vecx

                # Kマトリクスのi行、i列を全て0にし、i行i列の値を1にする
                matKtc[:, i] = 0.0
                matKtc[i, :] = 0.0
                matKtc[i, i] = 1.0
        matKtc = matKtc.tocsr()

        for i in range(len(vecBoundDisp)):
            if not vecBoundDisp[i] == None:
                vecRc[i] = vecBoundDisp[i] - vecDisp[i]

        return matKtc, vecRc

    # 全ての要素内の変数を更新する
    # vecDisp : 全節点の変位ベクトル(np.array型)
    # incNo   : インクリメントの番号
    def updateElements(self, vecDisp, incNo):

        for elem in self.elements:
            vecElemDisp = np.zeros(len(elem.nodes) * self.nodeDof)
            for i in range(len(elem.nodes)):
                for j in range(elem.nodeDof):
                    vecElemDisp[i * elem.nodeDof + j] = vecDisp[(elem.nodes[i].no - 1) * self.nodeDof + j]
            elem.update(vecElemDisp, incNo) 

    # 内力ベクトルを作成する
    def makeVecQ(self):
            vecQ = np.zeros(len(self.nodes) * self.nodeDof)
            for elem in self.elements:
                vecQe = elem.makeQeVector()
                for k in range(len(elem.nodes)):
                    for l in range(elem.nodeDof):
                        vecQ[(elem.nodes[k].no - 1) * self.nodeDof + l] += vecQe[k * elem.nodeDof + l]
            return vecQ

    # 解析結果をテキストファイルに出力する
    def outputTxt(self, filePath):

        # ファイルを作成し、開く
        f = open(filePath + ".txt", 'w')

        # 出力する文字の情報を定義する
        columNum = 20
        floatDigits = ".10g"

        # 入力データのタイトルを書きこむ
        f.write("*********************************\n")
        f.write("*          Input Data           *\n")
        f.write("*********************************\n")
        f.write("\n")

        # 節点情報を出力する
        f.write("***** Node Data ******\n")
        f.write("No".rjust(columNum) + "X".rjust(columNum) + "Y".rjust(columNum) + "Z".rjust(columNum) + "\n")
        f.write("-" * columNum * 4 + "\n")
        for node in self.nodes:
            strNo = str(node.no).rjust(columNum)
            strX = str(format(node.x, floatDigits).rjust(columNum))
            strY = str(format(node.y, floatDigits).rjust(columNum))
            strZ = str(format(node.z, floatDigits).rjust(columNum))
            f.write(strNo + strX + strY + strZ + "\n")
        f.write("\n")

        # 要素情報を出力する
        nodeNoColumNum = 36
        f.write("***** Element Data ******\n")
        f.write("No".rjust(columNum) + "Type".rjust(columNum) + "Node No".rjust(nodeNoColumNum) + "Density".rjust(columNum) + "\n")
        f.write("-" * columNum * 7 + "-" * nodeNoColumNum + "\n")
        for elem in self.elements:
            strNo = str(elem.no).rjust(columNum)
            strType = str(elem.__class__.__name__ ).rjust(columNum)
            strNodeNo = ""
            for node in elem.nodes:
                strNodeNo += " " + str(node.no)
            strNodeNo = strNodeNo.rjust(nodeNoColumNum)
            strDensity = "None".rjust(columNum)
            if not elem.density is None:
                strDensity = str(format(elem.density, floatDigits).rjust(columNum))
            f.write(strNo + strType + strNodeNo + strDensity + "\n")
        f.write("\n")

        # 単点拘束情報を出力する
        f.write("***** SPC Constraint Data ******\n")
        f.write("NodeNo".rjust(columNum) + "X Displacement".rjust(columNum) + "Y Displacement".rjust(columNum) + 
                "Z Displacement".rjust(columNum) + "\n")
        f.write("-" * columNum * 4 + "\n")
        vecBoundDisp = self.bound.makeDispVector()
        for i in range(self.bound.nodeNum):
            strFlg = False
            for j in range(self.bound.nodeDof):
                if not vecBoundDisp[i * self.bound.nodeDof + j] is None:
                    strFlg = True
            if strFlg == True:
                strNo = str(i + 1).rjust(columNum)
                strXDisp = "None".rjust(columNum)
                if not vecBoundDisp[i * self.bound.nodeDof] is None:
                    strXDisp = str(format(vecBoundDisp[i * self.bound.nodeDof], floatDigits).rjust(columNum))
                strYDisp = "None".rjust(columNum)
                if not vecBoundDisp[i * self.bound.nodeDof + 1] is None:
                    strYDisp = str(format(vecBoundDisp[i * self.bound.nodeDof + 1], floatDigits).rjust(columNum))
                strZDisp = "None".rjust(columNum)
                if not vecBoundDisp[i * self.bound.nodeDof + 2] is None:
                    strZDisp = str(format(vecBoundDisp[i * self.bound.nodeDof + 2], floatDigits).rjust(columNum))
                f.write(strNo + strXDisp + strYDisp + strZDisp + "\n")
        f.write("\n")

        # 荷重条件を出力する(等価節点力も含む)
        f.write("***** Nodal Force Data ******\n")
        f.write("NodeNo".rjust(columNum) + "X Force".rjust(columNum) + "Y Force".rjust(columNum) + 
                "Z Force".rjust(columNum) + "\n")
        f.write("-" * columNum * 4 + "\n")
        vecf = self.makeForceVector()
        for i in range(len(self.nodes)):
            strFlg = False
            for j in range(self.bound.nodeDof):
                if not vecf[i * self.bound.nodeDof + j] == 0.0:
                    strFlg = True
            if strFlg == True:
                strNo = str(i + 1).rjust(columNum)
                strXForce = str(format(vecf[i * self.bound.nodeDof], floatDigits).rjust(columNum))
                strYForce = str(format(vecf[i * self.bound.nodeDof + 1], floatDigits).rjust(columNum))
                strZForce = str(format(vecf[i * self.bound.nodeDof + 2], floatDigits).rjust(columNum))
                f.write(strNo + strXForce + strYForce + strZForce + "\n")
        f.write("\n")

        # 結果データのタイトルを書きこむ
        f.write("**********************************\n")
        f.write("*          Result Data           *\n")
        f.write("**********************************\n")
        f.write("\n")

        for i in range(self.incNum):
            f.write("*Increment " + str(i + 1) + "\n")
            f.write("\n")

            # 変位のデータを出力する
            f.write("***** Displacement Data ******\n")
            f.write("NodeNo".rjust(columNum) + "Magnitude".rjust(columNum) + "X Displacement".rjust(columNum) +
                    "Y Displacement".rjust(columNum) + "Z Displacement".rjust(columNum) + "\n")
            f.write("-" * columNum * 5 + "\n")
            for j in range(len(self.nodes)):
                strNo = str(j + 1).rjust(columNum)
                vecDisp = self.vecDispList[i]
                mag = np.linalg.norm(np.array((vecDisp[self.nodeDof * j], vecDisp[self.nodeDof * j + 1], vecDisp[self.nodeDof * j + 2])))
                strMag = str(format(mag, floatDigits).rjust(columNum))
                strXDisp = str(format(vecDisp[self.nodeDof * j], floatDigits).rjust(columNum))
                strYDisp = str(format(vecDisp[self.nodeDof * j + 1], floatDigits).rjust(columNum))
                strZDisp = str(format(vecDisp[self.nodeDof * j + 2], floatDigits).rjust(columNum))
                f.write(strNo + strMag + strXDisp + strYDisp + strZDisp + "\n")            
            f.write("\n")

            # 応力データを出力する
            f.write("***** Stress Data ******\n")
            f.write("Element No".rjust(columNum) + "Integral No".rjust(columNum) + "Stress XX".rjust(columNum) + "Stress YY".rjust(columNum) + 
                    "Stress ZZ".rjust(columNum) + "Stress XY".rjust(columNum) + "Stress XZ".rjust(columNum) + "Stress YZ".rjust(columNum) +
                    "Mises".rjust(columNum) + "\n")
            f.write("-" * columNum * 9 + "\n")
            for elemOutputData in self.elemOutputDataList[i]:
                elem = elemOutputData.element
                strElemNo = str(elem.no).rjust(columNum)
                for j in range(elem.ipNum):
                    strIntNo = str(j + 1).rjust(columNum)
                    strStressXX = str(format(elemOutputData.vecIpStressList[j][0], floatDigits).rjust(columNum))
                    strStressYY = str(format(elemOutputData.vecIpStressList[j][1], floatDigits).rjust(columNum))
                    strStressZZ = str(format(elemOutputData.vecIpStressList[j][2], floatDigits).rjust(columNum))
                    strStressXY = str(format(elemOutputData.vecIpStressList[j][3], floatDigits).rjust(columNum))
                    strStressXZ = str(format(elemOutputData.vecIpStressList[j][4], floatDigits).rjust(columNum))
                    strStressYZ = str(format(elemOutputData.vecIpStressList[j][5], floatDigits).rjust(columNum))
                    strMises = str(format(elemOutputData.ipMiseses[j], floatDigits).rjust(columNum))
                    f.write(strElemNo + strIntNo + strStressXX + strStressYY + strStressZZ + 
                            strStressXY + strStressXZ + strStressYZ + strMises + "\n")
            f.write("\n")

            # 第2ピオラ・キルヒホッフ応力データを出力する
            f.write("***** 2PK Stress Data ******\n")
            f.write("Element No".rjust(columNum) + "Integral No".rjust(columNum) + "2PKStress XX".rjust(columNum) + "2PKStress YY".rjust(columNum) + 
                    "2PKStress ZZ".rjust(columNum) + "2PKStress XY".rjust(columNum) + "2PKStress XZ".rjust(columNum) + "2PKStress YZ".rjust(columNum) + "\n")
            f.write("-" * columNum * 9 + "\n")
            for elemOutputData in self.elemOutputDataList[i]:
                elem = elemOutputData.element
                strElemNo = str(elem.no).rjust(columNum)
                for j in range(elem.ipNum):
                    strIntNo = str(j + 1).rjust(columNum)
                    str2PKStressXX = str(format(elemOutputData.vecIp2PKStressList[j][0], floatDigits).rjust(columNum))
                    str2PKStressYY = str(format(elemOutputData.vecIp2PKStressList[j][1], floatDigits).rjust(columNum))
                    str2PKStressZZ = str(format(elemOutputData.vecIp2PKStressList[j][2], floatDigits).rjust(columNum))
                    str2PKStressXY = str(format(elemOutputData.vecIp2PKStressList[j][3], floatDigits).rjust(columNum))
                    str2PKStressXZ = str(format(elemOutputData.vecIp2PKStressList[j][4], floatDigits).rjust(columNum))
                    str2PKStressYZ = str(format(elemOutputData.vecIp2PKStressList[j][5], floatDigits).rjust(columNum))
                    f.write(strElemNo + strIntNo + str2PKStressXX + str2PKStressYY + str2PKStressZZ + 
                            str2PKStressXY + str2PKStressXZ + str2PKStressYZ + "\n")
            f.write("\n") 

            # グリーン・ラグランジュひずみデータを出力する
            f.write("***** Green-Lagrange Strain Data ******\n")
            f.write("Element No".rjust(columNum) + "Integral No".rjust(columNum) + "GLStrain XX".rjust(columNum) + "GLStrain YY".rjust(columNum) + 
                    "GLStrain ZZ".rjust(columNum) + "GLStrain XY".rjust(columNum) + "GLStrain XZ".rjust(columNum) + "GLStrain YZ".rjust(columNum) + "\n")
            f.write("-" * (columNum * 8 + 30) + "\n")
            for elemOutputData in self.elemOutputDataList[i]:
                elem = elemOutputData.element
                strElemNo = str(elem.no).rjust(columNum)
                for j in range(elem.ipNum):
                    strIntNo = str(j + 1).rjust(columNum)
                    strGLStrainXX = str(format(elemOutputData.vecIpGLEStrainList[j][0], floatDigits).rjust(columNum))
                    strGLStrainYY = str(format(elemOutputData.vecIpGLEStrainList[j][1], floatDigits).rjust(columNum))
                    strGLStrainZZ = str(format(elemOutputData.vecIpGLEStrainList[j][2], floatDigits).rjust(columNum))
                    strGLStrainXY = str(format(elemOutputData.vecIpGLEStrainList[j][3], floatDigits).rjust(columNum))
                    strGLStrainXZ = str(format(elemOutputData.vecIpGLEStrainList[j][4], floatDigits).rjust(columNum))
                    strGLStrainYZ = str(format(elemOutputData.vecIpGLEStrainList[j][5], floatDigits).rjust(columNum))                    
                    f.write(strElemNo + strIntNo + strGLStrainXX + strGLStrainYY + strGLStrainZZ + 
                            strGLStrainXY + strGLStrainXZ + strGLStrainYZ + "\n")
            f.write("\n")          

            # 反力のデータを出力する
            f.write("***** Reaction Force Data ******\n")
            f.write("NodeNo".rjust(columNum) + "Magnitude".rjust(columNum) + "X Force".rjust(columNum) + "Y Force".rjust(columNum) + 
                    "Z Force".rjust(columNum) + "\n")
            f.write("-" * columNum * 5 + "\n")
            for j in range(len(self.nodes)):
                strNo = str(j + 1).rjust(columNum)
                vecRF = self.vecRFList[i]
                mag = np.linalg.norm(np.array((vecRF[self.nodeDof * j], vecRF[self.nodeDof * j + 1], vecRF[self.nodeDof * j + 2])))
                strMag = str(format(mag, floatDigits).rjust(columNum))
                strXForce = str(format(vecRF[self.nodeDof * j], floatDigits).rjust(columNum))
                strYForce = str(format(vecRF[self.nodeDof * j + 1], floatDigits).rjust(columNum))
                strZForce = str(format(vecRF[self.nodeDof * j + 2], floatDigits).rjust(columNum))
                f.write(strNo + strMag + strXForce + strYForce + strZForce + "\n")            
            f.write("\n")

        # ファイルを閉じる
        f.close()

    # 解析結果をvtkファイルに出力する
    def outputVTK(self, filePath):
        outputVTK = OutputVTK(self.nodes, self.elements, self.vecDispList, self.vecRFList, self.elemOutputDataList, self.incNum)
        outputVTK.output(filePath)