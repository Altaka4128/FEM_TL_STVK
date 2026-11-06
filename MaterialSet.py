class MaterialSet:

    # コンストラクタ
    # name     : 名前
    # density  : 密度
    # material : 材料の構成則
    def __init__(self, name, density = None, material = None):
        self.name = name
        self.density = density
        self.material = material