class Clock:
    def __init__(self,hours=0):
        self.hours = hours
    def display(self):
        print("hours : %d" % (self.hours))
class SubClock(Clock):
    def __init__(self, hours = 1, minute = 2):
        Clock.__init__(self,hours)
        self.minute = minute
    def display(self):
        print("hours: %d, minute: %d" % (self.hours, self.minute))
x = Clock()
x.display()
y = SubClock()
y.display()
