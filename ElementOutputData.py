import copy as cp

# 要素の出力データを格納するクラス
class ElementOutputData:

    # コンストラクタ
    # element        : 要素のクラス

    def __init__(self, element):

        # インスタンス変数を定義する
        self.element = cp.copy(element)

        # 積分点の応力、ひずみを計算する
        self.vecIpStressList = element.vecStressList.copy()         # 積分点の応力ベクトルのリスト(np.arrayのリスト型)
        self.ipMiseses = element.misesList.copy()                   # 積分点のミーゼス応力
        self.vecIpGLEStrainList = element.vecGLEStrainList.copy()   # 積分点のグリーン・ラグランジュひずみベクトルのリスト(np.arrayのリスト型)
        self.vecIp2PKStressList = element.vec2PKStressList.copy()   # 積分点の第2ピオラ・キルヒホッフ応力ベクトルのリスト(np.arrayのリスト型)
        self.centMises          = element.centMises.copy()          # 要素中心のミーゼス応力
