class A:
    def __init__(self,name=""):
        self.name = name
    def display(self):
        print("name : %s" % (self.name))
    def changeName(self):
        pass
class B(A):
    def __init__(self, name=""):
        A.__init__(self,name)
    def changeName(self):
        self.name += 'B class'
class C(A):
    def __init__(self, name=""):
        A.__init__(self,name)
    def changeName(self):
        self.name += 'C class'

b = B()
b.changeName()
b.display()
c = C()
c.changeName()
c.display()
