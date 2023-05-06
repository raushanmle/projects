from logging import disable
import torch as tr
import pandas as pd
from torch._C import Module

df = pd.read_csv(r"C:\Users\Raushan\Downloads\Code\wine-m\train.csv")

df.drop('id', inplace = True, axis=1)


class Net(tr.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = 


class raushan:
    leave = 9
    def __init__(self, name, age, salary):
        self.name = name
        self.age = age
        self.salary = salary
    def print_det(self):
        print(self.name, self.age, self.salary)
    @classmethod
    def change_leave(cls, leave_no):
        cls.leave = leave_no
    @classmethod
    def strin_split(cls, string_mtd):
        splitted_text = string_mtd.split("-")
        return cls(splitted_text[0], splitted_text[1], splitted_text[2])

    @staticmethod

    def just_fn(arg1):
        return arg1.split("-")


fn = raushan('raushan', 24, 20)

fn.change_leave(5)
fn.leave

fn2 = raushan.strin_split('rahul-34-50')
fn2.name

raushan.just_fn('rahul-34-50')


class programmer(raushan):
    def __init__(self, name, age, salary, program):
        self.name = name
        self.age = age
        self.salary = salary
        self.program = program

    def print_det(self):
        print(self.name, self.age, self.salary,self.program)


raush = programmer('raushan', 33, 333,'python')
raush.print_det()

class player:
    game = 'cod1'
    def __init__(self, name, game):
        self.name = name
        self.game = game

    def print_det(self):
        print(self.name, self.game)


class programmer1(programmer):
    def __init__(self, name, age, salary, program, exp):
        self.name = name
        self.age = age
        self.salary = salary
        self.program = program
        self.exp = exp

    def print_det1(self):
        print(self.name, self.age, self.salary,self.program, self.exp)

cls_cls = programmer1('rauha', 34, 335, 'python','4')
cls_cls.print_det1()

raus = player('raush', 'cod')
raus.print_det()

class cool_prgrammer(raushan, player):
    laguage1 = 'python3'
    def print_all(self):
        print(self.laguage1,self.game)

two_class = cool_prgrammer('raushan', 34, 345)
two_class.print_all()


class mobile:
    type1 = 'featurephone'
    def __init__(self, display, battery):
        self.display = display
        self.battery = battery
    def specs(self):
        print('_____________')
        print('your phone is',self.type1, 'specs are', self.display, self.battery)

mobile1 = mobile('TFT', 1000)
mobile1.specs()

class smartphone(mobile):
    type1 = 'smartphone'
    def __init__(self, display, battery, ram):
        self.display = display
        self.battery = battery
        self.ram = ram
    def specs(self):
        print('_____________')
        print('your phone is',self.type1, 'specs are', self.display, self.battery, self.ram)

phone = smartphone('amoled', 3000, 4)
phone.specs()

class smartphone2(mobile):
    def __init__(self, display, battery, ram):
        self.display = display
        self.battery = battery
        self.ram = ram
    def specs(self):
        print('_____________')
        print('your phone is',self.type1, 'specs are', self.display, self.battery, self.ram)

phone1 = smartphone('amoled', 5000, 6)
phone1.specs()


class mobile1:
    type1 = 'featurephone'
    def __init__(self):
        self.display = 'm1_var'
        self.battery = 'm1_var2'
    def specs(self):
        #print('_____________')
        print('your phone is',self.type1, 'specs are', self.display, self.battery)

class smartphone3(mobile1):
    type1 = 'smartphone_adv'
    def __init__(self, connectivity):
        self.connectivity = connectivity
        super().__init__()
    def specs1(self):
        super().specs()
        print(self.connectivity)
        

phone1 = mobile1()
phone1.display

phone2 = smartphone3('4g')
phone2.specs()
phone2.specs1()



class mobile2:
    type1 = 'featurephone'
    def __init__(self, display, battery):
        self.display = display
        self.battery = battery
    def specs(self):
        #print('_____________')
        print('your phone is',self.type1, 'specs are', self.display, self.battery)
    def use_of_init(self, display1, ram):
        print('self display', self.display)
        print('ram', ram)
        print('init display',display1)
    @classmethod
    def use_of_class(cls, connectivity):
        print("use of class", cls.type1)
        print('connectivity', connectivity)
    @staticmethod
    def use_of_static(var1, var2):
        print(var1+var2)

    def compare(self):
        return self.display + 'compare deco'

    def using_compare(self):
        return self.compare() + 'used compare'

        






mobile2()



mobile2('TFT', 3000).use_of_static('sita', 'ram')
mobile2('TFT', 3000).use_of_class('5g')
mobile2('TFT', 3000).use_of_init('amoled',4)
mobile2('TFT', 3000).using_compare()

class mobile1:
    type1 = 'featurephone'
    def __init__(self):
        self.display = 'm1_var'
        self.battery = 'm1_var2'
    def specs(self):
        #print('_____________')
        print('your phone is',self.type1, 'specs are', self.display, self.battery)

class smartphone3(mobile1):
    type1 = 'smartphone_adv'
    def __init__(self, connectivity):
        self.connectivity = connectivity
        super().__init__()

    def specs1(self):
        #super().specs()
        print(super().battery)
        #print(self.connectivity)

smartphone3('4g').battery


class A:
    classvar1 = 'I am a class variable in class A'
    def __init__(self):
        self.var1 = 'i am inside a class cons'
        self.classvar1 = 'inside var in class a'
        self.special = 'special'

class B(A):
    classvar1 = 'I am in class B'
    def __init__(self):
        print('exce')
        super(B, self).__init__()
        self.var1 = 'i am inside b cons'
        self.classvar1 = 'instance var in cls b'

    def check_class(self):
        print('check class')
        print(self.var1)

a = A()
b = B()

b.special
b.check_class()


class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width

# Here we declare that the Square class inherits from the Rectangle class
class Square(Rectangle):
    def __init__(self, length):
        super().__init__(length, length)

class Triangle:
    def __init__(self, base, height):
        self.base = base
        self.height = height
        super().__init__()

    def tri_area(self):
        return 0.5 * self.base * self.height

class RightPyramid(Square, Triangle):
    def __init__(self, base, slant_height):
        self.base = base
        self.slant_height = slant_height
        super().__init__(self.base)

    def area(self):
        base_area = super().area()
        
        perimeter = super().perimeter()
        return 0.5 * perimeter * self.slant_height + base_area

    def area_2(self):
        base_area = super().area()
        print(base_area)
        triangle_area = super().tri_area()
        return triangle_area * 4 + base_area


pyramid = RightPyramid(base=2, slant_height=4)
pyramid.area()
pyramid.area_2()


square = Square(4)
square.area()

class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width

class Square(Rectangle):
    def __init__(self, length):
        super().__init__(length, length)

class Cube(Square):
    def surface_area(self):
        face_area = super().area()
        return face_area * 6

    def pari(self):
        print(super().perimeter())

    def par2(self):
        print(super().perimeter())

    def volume(self):
        face_area = super().area()
        return face_area * self.length


Cube(4).par2()


class raushan:
    def __init__(self, name):
        self.name = name

class raush(raushan):
    def __init__(self, name):
        super(raush, self).__init__(name)

    def print_raush(self, last):
        print(self.name, last)

raush('raush').print_raush('kumar')


import torch
from torch import nn 

x = torch.tensor([[1], [2], [3], [4]], dtype= torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype= torch.float32)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.lin(x)


model = LinearRegression(1, 1)
model = nn.Linear(1, 1)

learning_rate = .01
n_iters = 100
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate)

for epoch in range(n_iters):
    y_pred = model(x)
    l = loss(y, y_pred)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (epoch + 1) % 10 == 0:
        print(epoch + 1, l.item())




predicted = model(x)


torch.sigmoid(x)
