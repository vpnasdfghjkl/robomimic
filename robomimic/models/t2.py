class D:
    def __init__(self) -> None:
        print("t: D")
class A:
    def __init__(self) -> None:
        print("here")
        
    def __init_subclass__(cls, **kwargs):
        print("sub class")
        
        
class B(A):
    def __init__(self, x) -> None:
        super().__init__(x)
    
class C(A):
    def prt():
        pass
