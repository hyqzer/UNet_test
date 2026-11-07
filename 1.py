import os

class A:
    def __init__(self, num):
        self.num = num

    def __call__(self, aa=None):
        if aa != None:
            return aa + self.num
        else:
            return self.num

    def __repr__(self):
        return f"这是一个 A 的实例"


a = A(2)

print(a())
print(a)