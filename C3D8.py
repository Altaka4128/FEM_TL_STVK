import numpy as np
import numpy.linalg as LA
from ElementOutputData import ElementOutputData

# 6面体8節点要素のクラス
class C3D8:
    # コンストラクタ
    # no              : 要素番号
    # nodes           : 節点の集合(Node型のリスト)
    # matSec        : # 材料情報(MaterialSet型)
    def __init__(self, no, nodes, matSet):
          
        # インスタンス変数を定義する
        self.nodeNum = 8                       # 節点の数
        self.nodeDof = 3                       # 節点の自由度
        self.no = no                           # 要素番号
        self.nodes = nodes                     # 節点の集合(Node型のリスト)
        self.density = matSet.density          # 密度
        self.material = matSet.material        # 材料の構成則
        self.ipNum = 8                         # 積分点の数
        self.w1 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # 積分点の重み係数1
        self.w2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # 積分点の重み係数2
        self.w3 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   # 積分点の重み係数3
        self.ai = np.array([-np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0),     # 積分点の座標(a,b,c座標系, np.array型のリスト)
                            -np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0)])
        self.bi = np.array([-np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0),     # 積分点の座標(a,b,c座標系, np.array型のリスト)
                            -np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0)])
        self.ci = np.array([-np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0), -np.sqrt(1.0 / 3.0),   # 積分点の座標(a,b,c座標系, np.array型のリスト)
                            np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0), np.sqrt(1.0 / 3.0)])
        self.matxiyizi = np.array([[self.nodes[0].x, self.nodes[0].y, self.nodes[0].z],
                                   [self.nodes[1].x, self.nodes[1].y, self.nodes[1].z],
                                   [self.nodes[2].x, self.nodes[2].y, self.nodes[2].z],
                                   [self.nodes[3].x, self.nodes[3].y, self.nodes[3].z],
                                   [self.nodes[4].x, self.nodes[4].y, self.nodes[4].z],
                                   [self.nodes[5].x, self.nodes[5].y, self.nodes[5].z],
                                   [self.nodes[6].x, self.nodes[6].y, self.nodes[6].z],
                                   [self.nodes[7].x, self.nodes[7].y, self.nodes[7].z]]) 
        self.incNo = 0   # インクリメントのNo

        # 要素内の各行列、ベクトルを初期化する       
        self.vecDisp = np.zeros(self.nodeNum * self.nodeDof)                # 要素内の変位
        self.vecStressList = [np.zeros(6) for i in range(self.ipNum)]       # 要素内のコーシー応力(np.array型のリスト)
        self.misesList = [0.0 for i in range(self.ipNum)]                   # 要素内のミーゼス応力(リスト)
        self.vecGLEStrainList = [np.zeros(6) for i in range(self.ipNum)]    # 要素内のグリーン・ラグランジュひずみ(np.array型のリスト)
        self.vec2PKStressList = [np.zeros(6) for i in range(self.ipNum)]    # 要素内の第2ピオラ・キルヒホッフ応力(np.array型のリスト)
        self.centMises = 0.0                                                # 要素中心のミーゼス応力
        self.matdNdabcList = [np.zeros((3,8)) for i in range(self.ipNum)]   # dNdabcマトリクス(np.array型のリスト)
        self.matdNdxyzList = [np.zeros((3,8)) for i in range(self.ipNum)]   # dNdxyzマトリクス(np.array型のリスト)
        self.matJList   = [np.zeros((3,3)) for i in range(self.ipNum)]      # ヤコビ行列(np.array型のリスト)
        self.matFList   = [np.zeros((3,3)) for i in range(self.ipNum)]      # 変形勾配テンソル(np.array型のリスト)
        self.matDList   = [np.zeros((6,6)) for i in range(self.ipNum)]      # Dマトリクス(np.array型のリスト)
        self.matBl0List = [np.zeros((6,24)) for i in range(self.ipNum)]     # BL0マトリクス(np.array型のリスト)
        self.matBl1List = [np.zeros((6,24)) for i in range(self.ipNum)]     # BL1マトリクス(np.array型のリスト)
        self.matBnlList = [np.zeros((9,24)) for i in range(self.ipNum)]     # BNLマトリクス(np.array型のリスト)
        self.matSList   = [np.zeros((9,9)) for i in range(self.ipNum)]      # Sマトリクス(np.array型のリスト)
        self.update(np.zeros(self.nodeNum * self.nodeDof), 0)

    # 要素接線剛性マトリクスKetを作成する
    def makeKetmatrix(self):
        
        # Ketマトリクスをガウス積分で計算する
        matKetl  = np.zeros([self.nodeDof * self.nodeNum, self.nodeDof * self.nodeNum])
        matKetnl = np.zeros([self.nodeDof * self.nodeNum, self.nodeDof * self.nodeNum])
        for i in range(self.ipNum):
            detMatJ = LA.det(self.matJList[i])
            matKetl  += self.w1[i] * self.w2[i] * self.w3[i] * \
                        (self.matBl0List[i] + self.matBl1List[i]).T @ self.matDList[i] @ \
                        (self.matBl0List[i] + self.matBl1List[i]) * detMatJ
            matKetnl += self.w1[i] * self.w2[i] * self.w3[i] * \
                        self.matBnlList[i].T @ self.matSList[i] @ self.matBnlList[i] * detMatJ
        matKet = matKetl + matKetnl
        
        return matKet

   # 要素内力ベクトルQeを作成する
    def makeQeVector(self):

        # 要素内力ベクトルQeを計算する
        vecQe = np.zeros(self.nodeDof * self.nodeNum)
        for i in range(self.ipNum):
            vecQe += self.w1[i] * self.w2[i] * self.w3[i] * \
                     (self.matBl0List[i] + self.matBl1List[i]).T @ self.vec2PKStressList[i] * LA.det(self.matJList[i])

        return vecQe

    # 要素内の変数を更新する
    # vecDisp : 要素節点の変位ベクトル(np.array型)
    # incNo   : インクリメントNo
    def update(self, vecDisp, incNo):

        self.vecDisp = vecDisp
        self.incNo = incNo
        matuvw = self.vecDisp.reshape(8,3)
        for i in range(self.ipNum):

            # 各行列、ベクトルを計算する
            if incNo == 0:
                self.matdNdabcList[i] = self.makedNdabc(self.ai[i], self.bi[i], self.ci[i])
                self.matJList[i]      = self.makeJmatrix(self.matdNdabcList[i])
                self.matdNdxyzList[i] = LA.solve(self.matJList[i], self.matdNdabcList[i])
                self.matBl0List[i]    = self.makeBl0matrix(self.matdNdxyzList[i])
                self.matBnlList[i]    = self.makeBnlmatrix(self.matdNdxyzList[i])
            matduvwdxyz = self.matdNdxyzList[i] @ matuvw
            self.matFList[i]         = self.makeFmatrix(matduvwdxyz)
            self.vecGLEStrainList[i] = self.makeGLEStrainVector(self.matFList[i])
            self.vec2PKStressList[i] = self.make2PKStressVector(self.matFList[i])
            self.vecStressList[i]    = self.makeStressVector(self.matFList[i], self.vec2PKStressList[i])
            self.misesList[i]        = self.misesStress(self.vecStressList[i])
            self.matDList[i]         = self.material.makeDmatrix(self.matFList[i])
            self.matBl1List[i]       = self.makeBl1matrix(self.matdNdxyzList[i], matduvwdxyz)
            self.matSList[i]         = self.makeSmatrix(self.vec2PKStressList[i])
        # 要素中心値を計算する
        if incNo == 0:
            matdNdabc = self.makedNdabc(0.0, 0.0, 0.0)
            matJ = self.makeJmatrix(matdNdabc)
            self.matdNdxyzCent = LA.solve(matJ, matdNdabc)
        matduvwdxyz = self.matdNdxyzCent @ matuvw
        matF             = self.makeFmatrix(matduvwdxyz) 
        vecCent2PKStress = self.make2PKStressVector(matF)
        vecCentStress    = self.makeStressVector(matF, vecCent2PKStress)
        self.centMises   = self.misesStress(vecCentStress)

    # 要素の出力データを作成する
    def makeOutputData(self):

        output = ElementOutputData(self)

        return output
        
    # ヤコビ行列を計算する
    def makeJmatrix(self, matdNdabc):

        # ヤコビ行列を計算する
        matJ = matdNdabc @ self.matxiyizi
        
        return matJ

    # Bl0マトリクスを作成する
    def makeBl0matrix(self, matdNdxyz):
        
        # Bl0マトリクスを計算する
        matBl0 = np.zeros((6, self.nodeNum * self.nodeDof))
        for i in range(self.nodeNum): 
            matTmp = np.array([[matdNdxyz[0, i], 0.0, 0.0],
                               [0.0, matdNdxyz[1, i], 0.0],
                               [0.0, 0.0, matdNdxyz[2, i]],
                               [matdNdxyz[1, i], matdNdxyz[0, i], 0.0],
                               [0.0, matdNdxyz[2, i], matdNdxyz[1, i]],
                               [matdNdxyz[2, i], 0.0, matdNdxyz[0, i]]])
            c = 3 * i 
            matBl0[:, c:c+3] = matTmp
            
        return matBl0

    # Bl1マトリクスを作成する
    def makeBl1matrix(self, matdNdxyz, matduvwdxyz):

        # Bl1マトリクスを計算する
        matBl1 = np.zeros((6, self.nodeNum * self.nodeDof))
        for i in range(self.nodeNum): 
            e11 = matduvwdxyz[0, 0] * matdNdxyz[0, i]
            e12 = matduvwdxyz[0, 1] * matdNdxyz[0, i]
            e13 = matduvwdxyz[0, 2] * matdNdxyz[0, i]
            e21 = matduvwdxyz[1, 0] * matdNdxyz[1, i]
            e22 = matduvwdxyz[1, 1] * matdNdxyz[1, i]
            e23 = matduvwdxyz[1, 2] * matdNdxyz[1, i]
            e31 = matduvwdxyz[2, 0] * matdNdxyz[2, i]
            e32 = matduvwdxyz[2, 1] * matdNdxyz[2, i]
            e33 = matduvwdxyz[2, 2] * matdNdxyz[2, i]
            e41 = matduvwdxyz[1, 0] * matdNdxyz[0, i] + matduvwdxyz[0, 0] * matdNdxyz[1, i]
            e42 = matduvwdxyz[1, 1] * matdNdxyz[0, i] + matduvwdxyz[0, 1] * matdNdxyz[1, i]
            e43 = matduvwdxyz[1, 2] * matdNdxyz[0, i] + matduvwdxyz[0, 2] * matdNdxyz[1, i]
            e51 = matduvwdxyz[1, 0] * matdNdxyz[2, i] + matduvwdxyz[2, 0] * matdNdxyz[1, i]
            e52 = matduvwdxyz[1, 1] * matdNdxyz[2, i] + matduvwdxyz[2, 1] * matdNdxyz[1, i]
            e53 = matduvwdxyz[1, 2] * matdNdxyz[2, i] + matduvwdxyz[2, 2] * matdNdxyz[1, i]
            e61 = matduvwdxyz[2, 0] * matdNdxyz[0, i] + matduvwdxyz[0, 0] * matdNdxyz[2, i]
            e62 = matduvwdxyz[2, 1] * matdNdxyz[0, i] + matduvwdxyz[0, 1] * matdNdxyz[2, i]
            e63 = matduvwdxyz[2, 2] * matdNdxyz[0, i] + matduvwdxyz[0, 2] * matdNdxyz[2, i]
                   
            matTmp = np.array([[e11, e12, e13],
                               [e21, e22, e23],
                               [e31, e32, e33],
                               [e41, e42, e43],
                               [e51, e52, e53],
                               [e61, e62, e63]])
            c = 3 * i 
            matBl1[:, c:c+3] = matTmp
            
        return matBl1

    # Bnlマトリクスを作成する
    def makeBnlmatrix(self, matdNdxyz):
        
        # Bnlマトリクスを計算する
        matBnl = np.zeros((9, self.nodeNum * self.nodeDof))
        for i in range(self.nodeNum): 
            matTmp = np.array([[matdNdxyz[0, i], 0.0, 0.0],
                               [0.0, matdNdxyz[0, i], 0.0],
                               [0.0, 0.0, matdNdxyz[0, i]],
                               [matdNdxyz[1, i], 0.0, 0.0],
                               [0.0, matdNdxyz[1, i], 0.0],
                               [0.0, 0.0, matdNdxyz[1, i]],
                               [matdNdxyz[2, i], 0.0, 0.0],
                               [0.0, matdNdxyz[2, i], 0.0],
                               [0.0, 0.0, matdNdxyz[2, i]]]) 
            c = 3 * i 
            matBnl[:, c:c+3] = matTmp
            
        return matBnl
    
    # Sマトリクスを作成する
    def makeSmatrix(self, vec2PKStress):

        # Sマトリクスを計算する
        matS   = np.zeros((9,9))
        matS11 = np.identity(3) * vec2PKStress[0]
        matS22 = np.identity(3) * vec2PKStress[1]
        matS33 = np.identity(3) * vec2PKStress[2]
        matS12 = np.identity(3) * vec2PKStress[3]
        matS23 = np.identity(3) * vec2PKStress[4]
        matS31 = np.identity(3) * vec2PKStress[5]
        matS[0:3, 0:3] = matS11
        matS[0:3, 3:6] = matS12
        matS[0:3, 6:9] = matS31 
        matS[3:6, 0:3] = matS12
        matS[3:6, 3:6] = matS22
        matS[3:6, 6:9] = matS23
        matS[6:9, 0:3] = matS31
        matS[6:9, 3:6] = matS23
        matS[6:9, 6:9] = matS33

        return matS

    # dNdabcの行列を計算する
    def makedNdabc(self, a, b, c):

        # dNi/da, dNi/db, dNi/dcを計算する
        dN1da = -0.125 * (1.0 - b) * (1.0 - c)
        dN2da = 0.125 * (1.0 - b) * (1.0 - c)
        dN3da = 0.125 * (1.0 + b) * (1.0 - c)
        dN4da = -0.125 * (1.0 + b) * (1.0 - c)
        dN5da = -0.125 * (1.0 - b) * (1.0 + c)
        dN6da = 0.125 * (1.0 - b) * (1.0 + c)
        dN7da = 0.125 * (1.0 + b) * (1.0 + c)
        dN8da = -0.125 * (1.0 + b) * (1.0 + c)
        dN1db = -0.125 * (1.0 - a) * (1.0 - c)
        dN2db = -0.125 * (1.0 + a) * (1.0 - c)
        dN3db = 0.125 * (1.0 + a) * (1.0 - c)
        dN4db = 0.125 * (1.0 - a) * (1.0 - c)
        dN5db = -0.125 * (1.0 - a) * (1.0 + c)
        dN6db = -0.125 * (1.0 + a) * (1.0 + c)
        dN7db = 0.125 * (1.0 + a) * (1.0 + c)
        dN8db = 0.125 * (1.0 - a) * (1.0 + c)
        dN1dc = -0.125 * (1.0 - a) * (1.0 - b)
        dN2dc = -0.125 * (1.0 + a) * (1.0 - b)
        dN3dc = -0.125 * (1.0 + a) * (1.0 + b)
        dN4dc = -0.125 * (1.0 - a) * (1.0 + b)
        dN5dc = 0.125 * (1.0 - a) * (1.0 - b)
        dN6dc = 0.125 * (1.0 + a) * (1.0 - b)
        dN7dc = 0.125 * (1.0 + a) * (1.0 + b)
        dN8dc = 0.125 * (1.0 - a) * (1.0 + b)

        # dNdabcを計算する
        dNdabc = np.array([[dN1da, dN2da, dN3da, dN4da, dN5da, dN6da, dN7da, dN8da],
                           [dN1db, dN2db, dN3db, dN4db, dN5db, dN6db, dN7db, dN8db],
                           [dN1dc, dN2dc, dN3dc, dN4dc, dN5dc, dN6dc, dN7dc, dN8dc]])
        
        return dNdabc

    # 変形勾配テンソルを作成する
    def makeFmatrix(self, matduvwdxyz):

        # 変形勾配テンソルを計算する
        matF = np.eye(3) + matduvwdxyz.T

        return matF

    # グリーン・ラグランジュひずみベクトルを作成する
    def makeGLEStrainVector(self, matF):

        # グリーン・ラグランジュひずみを計算する
        matGLEStrain = 0.5 * (matF.T @ matF - np.eye(3))
        vecGLEStrain = np.array([matGLEStrain[0,0], matGLEStrain[1,1], matGLEStrain[2,2], 2.0 * matGLEStrain[0,1], 2.0 * matGLEStrain[1,2], 2.0 * matGLEStrain[0,2]])

        return vecGLEStrain

    # 第2ピオラ・キルヒホッフ応力ベクトルを作成する
    def make2PKStressVector(self, matF):

        # グリーン・ラグランジュひずみを計算する
        vec2PKStress = self.material.make2PKStressVector(matF)

        return vec2PKStress

    # コーシー応力ベクトルを作成する
    def makeStressVector(self, matF, vec2PKStress):

        # コーシー応力ベクトルを計算する
        S11 = vec2PKStress[0]
        S22 = vec2PKStress[1]
        S33 = vec2PKStress[2]
        S12 = vec2PKStress[3]
        S23 = vec2PKStress[4]
        S13 = vec2PKStress[5]
        mat2PKStress = np.array([[S11, S12, S13],
                                 [S12, S22, S23],
                                 [S13, S23, S33]])
        matStress = matF @ mat2PKStress @ matF.T / LA.det(matF)
        vecStress = np.array([matStress[0, 0], matStress[1, 1], matStress[2, 2], matStress[0, 1], matStress[1, 2], matStress[0,2]])

        return vecStress

    # ミーゼス応力を計算する
    # vecStress : 応力ベクトル(np.array型)
    def misesStress(self, vecStress):

        tmp1 = np.square(vecStress[0] - vecStress[1]) + np.square(vecStress[1] - vecStress[2]) + np.square(vecStress[2] - vecStress[0])
        tmp2 = 6.0 * (np.square(vecStress[3]) + np.square(vecStress[4]) + np.square(vecStress[5]))
        mises = np.sqrt(0.5 * (tmp1 + tmp2))

        return mises

    # 体積を求める
    def getVolume(self):
        
        # ガウス積分で体積を計算する
        volume = 0
        for i in range(self.ipNum):
            volume += self.w1[i] * self.w2[i] * self.w3[i] * LA.det(self.matJList[i])

        return volume