import numpy as np
import numpy.linalg as LA

# サンブナン・キルヒホッフモデルの構成則を計算するためのクラス
class StVKMaterial:
    # コンストラクタ
    # young   : パラメータ1
    # poisson : パラメータ2
    def __init__(self, young, poisson):

        # インスタンス変数を定義する
        self.young   = young
        self.poisson = poisson

    # Dマトリクスを作成する
    # matF : 変形勾配テンソル
    def makeDmatrix(self, matF):
        
        tmp = self.young / ((1.0 + self.poisson) * (1.0 - 2.0 * self.poisson))
        matD = np.array([[1.0 - self.poisson, self.poisson, self.poisson, 0.0, 0.0, 0.0],
                         [self.poisson, 1.0 - self.poisson, self.poisson, 0.0, 0.0, 0.0],
                         [self.poisson, self.poisson, 1.0 - self.poisson, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * self.poisson), 0.0, 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * self.poisson), 0.0],
                         [0.0, 0.0, 0.0, 0.0, 0.0, 0.5 * (1.0 - 2.0 * self.poisson)]])
        matD = tmp * matD

        return matD

    # 第2ピオラ・キルヒホッフ応力を作成する
    # matF : 変形勾配テンソル
    def make2PKStressVector(self, matF):

        # グリーン・ラグランジュひずみを計算する
        matGLEStrain = 0.5 * (matF.T @ matF - np.eye(3))
        vecGLEStrain = np.array([matGLEStrain[0,0], matGLEStrain[1,1], matGLEStrain[2,2], 
                                 2.0 * matGLEStrain[0,1], 2.0 * matGLEStrain[1,2], 2.0 * matGLEStrain[0,2]])

        # 第2ピオラ・キルヒホッフ応力を計算する
        matD = self.makeDmatrix(matF)
        vec2PKStress = matD @ vecGLEStrain

        return vec2PKStress
